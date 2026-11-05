"""Shared logic for executors that keep one long-lived worker process per session."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .base import ExecResult, validate_session_id

Snapshot = dict[str, tuple[int, int]]
EOF = ""  # a real protocol line always ends in a newline, so "" is a safe end marker
OVERSIZE = "\0oversize"  # put by the pump when a line exceeds the size limit
QUEUED_LINES = 4  # the protocol has one reply per request; extra lines are dropped
MESSAGE_ROOM = 256  # room above max_output_chars for the worker's truncation note and errors


@dataclass
class _Session:
    workspace: Path
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: subprocess.Popen | None = None
    lines: queue.Queue | None = None


def snapshot_workspace(workspace: Path) -> Snapshot:
    """Map relative file path to (mtime_ns, size) for regular files. Hidden entries and
    symbolic links are skipped, so a link to a file outside the workspace is never reported."""
    state: Snapshot = {}
    for directory, subdirectories, names in os.walk(workspace, followlinks=False):
        subdirectories[:] = [d for d in subdirectories if not d.startswith(".")]
        for name in names:
            if name.startswith("."):
                continue
            path = Path(directory, name)
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                relative = path.relative_to(workspace).as_posix()
                state[relative] = (info.st_mtime_ns, info.st_size)
    return state


def changed_files(before: Snapshot, after: Snapshot) -> tuple[str, ...]:
    return tuple(sorted(name for name, meta in after.items() if before.get(name) != meta))


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... [{omitted} characters truncated]"


class WorkerExecutor(ABC):
    """Template for executors built on the JSON line protocol in worker.py.

    Subclasses provide `_spawn`. Calls on one session are serialized; different sessions run
    in parallel. A timeout or a worker crash discards that session's interpreter state.
    """

    def __init__(
        self,
        *,
        workspace_root: Path | str | None = None,
        timeout_s: float = 30.0,
        cpu_s: float = 30.0,
        memory_mb: int = 2048,
        max_output_chars: int = 20_000,
        startup_timeout_s: float = 30.0,
    ) -> None:
        self.timeout_s = timeout_s
        self.cpu_s = cpu_s
        self.memory_mb = memory_mb
        self.max_output_chars = max_output_chars
        self.startup_timeout_s = startup_timeout_s
        self._owns_root = workspace_root is None
        self._root = Path(workspace_root or tempfile.mkdtemp(prefix="cia-")).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, _Session] = {}
        self._registry_lock = threading.Lock()

    @abstractmethod
    def _spawn(self, session_id: str, workspace: Path) -> subprocess.Popen:
        """Start a worker with stdin and stdout pipes and return it."""

    def _terminate(self, session_id: str, process: subprocess.Popen) -> None:
        process.kill()

    def _session(self, session_id: str) -> _Session:
        validate_session_id(session_id)
        with self._registry_lock:
            session = self._sessions.get(session_id)
            if session is None:
                workspace = self._root / session_id
                workspace.mkdir(parents=True, exist_ok=True)
                if workspace.is_symlink() or workspace.resolve().parent != self._root:
                    raise ValueError(f"Workspace for session {session_id!r} is a symbolic link")
                session = self._sessions[session_id] = _Session(workspace)
            return session

    def workspace_path(self, session_id: str) -> Path:
        return self._session(session_id).workspace

    def run(self, session_id: str, code: str) -> ExecResult:
        session = self._session(session_id)
        started = time.monotonic()
        with session.lock:
            before = snapshot_workspace(session.workspace)
            result = self._run_locked(session_id, session, code)
            files = changed_files(before, snapshot_workspace(session.workspace))
        return ExecResult(
            stdout=truncate(result["stdout"], self.max_output_chars + MESSAGE_ROOM),
            stderr=truncate(result["stderr"], self.max_output_chars + MESSAGE_ROOM),
            ok=result["ok"],
            files=files,
            duration_s=round(time.monotonic() - started, 3),
        )

    def _run_locked(self, session_id: str, session: _Session, code: str) -> dict:
        if session.process is None or session.process.poll() is not None:
            failure = self._start(session_id, session)
            if failure:
                return {"stdout": "", "stderr": failure, "ok": False}
        request_id = uuid.uuid4().hex
        request = {
            "id": request_id,
            "code": code,
            "cpu_s": self.cpu_s,
            "max_output_chars": self.max_output_chars,
        }
        try:
            session.process.stdin.write(json.dumps(request) + "\n")
            session.process.stdin.flush()
        except OSError:
            return self._discard(session_id, session, None)
        reply = self._read_line(session, self.timeout_s)
        if reply is None:
            limit = f"Execution exceeded the {self.timeout_s:g} second time limit."
            return self._discard(session_id, session, limit)
        if reply == EOF:
            return self._discard(session_id, session, None)
        if reply == OVERSIZE:
            return self._discard(session_id, session, "Worker reply exceeded the size limit.")
        response = _parse_reply(reply, request_id)
        if response is None:
            return self._discard(session_id, session, "Worker sent an invalid protocol reply.")
        return response

    def _start(self, session_id: str, session: _Session) -> str:
        session.process = self._spawn(session_id, session.workspace)
        session.lines = queue.Queue(maxsize=QUEUED_LINES)
        threading.Thread(
            target=_pump,
            args=(session.process.stdout, session.lines, self._max_line_chars()),
            daemon=True,
        ).start()
        ready = self._read_line(session, self.startup_timeout_s)
        if ready is None or ready == EOF or not _is_ready(ready):
            details = self._startup_details(session_id)
            reason = "Worker failed to start." + (f" {details}" if details else "")
            return self._discard(session_id, session, reason)["stderr"]
        return ""

    def _startup_details(self, session_id: str) -> str:
        """Diagnostic text for a worker that failed to start. Subclasses may override."""
        return ""

    def _max_line_chars(self) -> int:
        # Two streams of max_output_chars, each character escaped to at most 12 JSON characters.
        return 24 * self.max_output_chars + 65_536

    def _read_line(self, session: _Session, timeout_s: float) -> str | None:
        """Next line from the worker, EOF if the worker closed its output, None on timeout."""
        try:
            return session.lines.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def _discard(self, session_id: str, session: _Session, reason: str | None) -> dict:
        """Kill the worker and describe why. A reason of None means the worker died by itself."""
        process = session.process
        self._terminate(session_id, process)
        status = process.wait()
        session.process = None
        session.lines = None
        if reason is None:
            reason = _describe_exit(status)
        return {
            "stdout": "",
            "stderr": f"{reason} The interpreter was restarted; earlier variables are gone.",
            "ok": False,
        }

    def reset(self, session_id: str) -> None:
        session = self._session(session_id)
        with session.lock:
            if session.process is not None:
                self._terminate(session_id, session.process)
                session.process.wait()
                session.process = None
                session.lines = None
            shutil.rmtree(session.workspace, ignore_errors=True)
            session.workspace.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        with self._registry_lock:
            sessions = dict(self._sessions)
            self._sessions.clear()
        for session_id, session in sessions.items():
            with session.lock:
                if session.process is not None:
                    self._terminate(session_id, session.process)
                    session.process.wait()
        if self._owns_root:
            shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> WorkerExecutor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _pump(stream, lines: queue.Queue, max_line_chars: int) -> None:
    """Move worker output lines to the queue. Stops at the first oversize line. Lines that do
    not fit in the queue are dropped; the reply check then rejects whatever arrives instead."""
    while True:
        line = stream.readline(max_line_chars)
        if line and not line.endswith("\n"):
            _offer(lines, OVERSIZE)
            break
        if not line:
            break
        _offer(lines, line)
    _offer(lines, EOF)


def _offer(lines: queue.Queue, item: str) -> None:
    with contextlib.suppress(queue.Full):
        lines.put_nowait(item)


def _is_ready(line: str) -> bool:
    try:
        return json.loads(line) == {"ready": True}
    except ValueError:
        return False


def _parse_reply(line: str, request_id: str) -> dict | None:
    """Return the reply if it is well formed and answers this request, otherwise None."""
    try:
        reply = json.loads(line)
    except ValueError:
        return None
    if (
        not isinstance(reply, dict)
        or reply.get("id") != request_id
        or not isinstance(reply.get("stdout"), str)
        or not isinstance(reply.get("stderr"), str)
        or not isinstance(reply.get("ok"), bool)
    ):
        return None
    return reply


def _describe_exit(status: int) -> str:
    if status < 0:
        name = signal.Signals(-status).name
        hint = {"SIGXCPU": " (CPU time limit)", "SIGKILL": " (memory limit or external kill)"}
        return f"Worker was terminated by {name}{hint.get(name, '')}."
    return f"Worker exited with status {status}."
