"""Persistent Python worker. Standalone on purpose: it uses only the standard library so the
docker executor can mount this single file into a stock Python image.

Protocol: one JSON object per line on stdin and stdout.
    worker -> parent   {"ready": true}
    parent -> worker   {"id": str, "code": str, "cpu_s": number, "max_output_chars": int}
    worker -> parent   {"id": str, "stdout": str, "stderr": str, "ok": bool}
After startup, file descriptors 0, 1 and 2 point at /dev/null, so printing cannot corrupt the
protocol; print output is captured in-process, up to max_output_chars per stream. Code that
writes to the protocol descriptors directly produces a reply the parent rejects, and the parent
then restarts the worker.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import linecache
import math
import os
import sys
import traceback

BLOCKED_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.forkpty",
        "ctypes.dlopen",
        "os.kill",
        "os.killpg",
    }
)
# Event name -> ((argument index, "read" or "write"), ...). A "write" path must resolve inside
# the workspace; a "read" path may also resolve inside the Python installation.
PATH_EVENTS = {
    "os.listdir": ((0, "read"),),
    "os.scandir": ((0, "read"),),
    "os.chdir": ((0, "read"),),
    "glob.glob": ((0, "read"),),
    "os.mkdir": ((0, "write"),),
    "os.rmdir": ((0, "write"),),
    "os.remove": ((0, "write"),),
    "os.rename": ((0, "write"), (1, "write")),
    "os.link": ((0, "write"), (1, "write")),
    "os.symlink": ((0, "read"), (1, "write")),
    "os.chmod": ((0, "write"),),
    "os.chown": ((0, "write"),),
    "os.truncate": ((0, "write"),),
    "os.utime": ((0, "write"),),
    "shutil.copyfile": ((0, "read"), (1, "write")),
    "shutil.copymode": ((0, "read"), (1, "write")),
    "shutil.copystat": ((0, "read"), (1, "write")),
    "shutil.copytree": ((0, "read"), (1, "write")),
    "shutil.move": ((0, "write"), (1, "write")),
    "shutil.rmtree": ((0, "write"),),
}
READ_ONLY_EXTRA_ROOTS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/etc/fonts",
    "/usr/share/zoneinfo",
    "/proc/self",
    "/proc/cpuinfo",
    "/proc/meminfo",
    "/sys/devices/system/cpu",
    "/sys/fs/cgroup",
    "/dev/null",
    "/dev/zero",
    "/dev/urandom",
    "/dev/random",
)
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _real(path: object) -> str | None:
    if isinstance(path, int) or path is None:
        return None
    try:
        return os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError):
        return None


def _inside(path: str, roots: list[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


class _LimitGate:
    """Lets the worker's own setrlimit calls through the audit hook and refuses all others."""

    open = False


def install_confinement(workspace: str) -> None:
    """Install an audit hook that refuses file access outside the workspace, process creation,
    signals to other processes and rlimit changes. This is a guardrail against mistakes, not a
    security boundary: native code and ordinary Python introspection can bypass it."""
    write_roots = (os.path.realpath(workspace),)
    read_roots = write_roots + tuple(
        os.path.realpath(p)
        for p in {
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            sys.base_exec_prefix,
            *[p for p in sys.path if p and os.path.isdir(p)],
            *READ_ONLY_EXTRA_ROOTS,
        }
    )
    roots_for = {"read": read_roots, "write": write_roots}
    real, inside, gate = _real, _inside, _LimitGate
    busy = False

    def hook(event: str, args: tuple) -> None:
        nonlocal busy
        if busy:
            return
        if event in BLOCKED_EVENTS:
            raise PermissionError(f"{event} is disabled in this sandbox")
        if event == "resource.setrlimit" and not gate.open:
            raise PermissionError("Changing resource limits is disabled in this sandbox")
        if event == "open":
            path = real(args[0])
            if path is None:
                return
            writing = isinstance(args[2], int) and args[2] & WRITE_FLAGS
            if not inside(path, write_roots if writing else read_roots):
                verb = "write" if writing else "read"
                raise PermissionError(f"Cannot {verb} outside the workspace: {args[0]!r}")
        elif event in PATH_EVENTS:
            busy = True
            try:
                for index, mode in PATH_EVENTS[event]:
                    path = real(args[index]) if index < len(args) else None
                    if path is not None and not inside(path, roots_for[mode]):
                        raise PermissionError(f"Path is outside the workspace: {args[index]!r}")
            finally:
                busy = False

    sys.addaudithook(hook)


def apply_limits(memory_mb: int) -> None:
    import resource

    if memory_mb > 0:
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    file_limit = 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))


def set_cpu_budget(seconds: float) -> None:
    """Allow `seconds` of additional CPU time. The soft limit raises SIGXCPU, which ends the
    worker; the parent reports that as a CPU limit violation."""
    import resource

    used = resource.getrusage(resource.RUSAGE_SELF)
    soft = int(used.ru_utime + used.ru_stime) + max(1, math.ceil(seconds))
    _LimitGate.open = True
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (soft, resource.RLIM_INFINITY))
    finally:
        _LimitGate.open = False


class BoundedWriter(io.TextIOBase):
    """Text sink that keeps the first `limit` characters and counts the rest."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._kept = 0
        self.omitted = 0

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        room = self._limit - self._kept
        if room > 0:
            self._parts.append(text[:room])
            self._kept += min(room, len(text))
        self.omitted += max(0, len(text) - max(room, 0))
        return len(text)

    def getvalue(self) -> str:
        text = "".join(self._parts)
        if self.omitted:
            text += f"\n... [{self.omitted} characters truncated]"
        return text


def _run_source(code: str, filename: str, namespace: dict) -> None:
    """Execute code; if the last statement is an expression, print its repr like a notebook."""
    tree = ast.parse(code, filename, "exec")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, filename, "exec"), namespace)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), filename, "eval"), namespace)
        if value is not None:
            print(repr(value))


def execute(code: str, namespace: dict, cell_number: int, max_output_chars: int) -> dict:
    filename = f"<cell {cell_number}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    out, err = BoundedWriter(max_output_chars), BoundedWriter(max_output_chars)
    ok = True
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            _run_source(code, filename, namespace)
        except BaseException as error:
            ok = False
            frame = error.__traceback__
            while frame is not None and not frame.tb_frame.f_code.co_filename.startswith("<cell"):
                frame = frame.tb_next
            traceback.print_exception(type(error), error, frame, file=err)
    return {"stdout": out.getvalue(), "stderr": err.getvalue(), "ok": ok}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--memory-mb", type=int, default=0)
    parser.add_argument("--no-confine", action="store_true")
    args = parser.parse_args(argv)

    protocol_in = os.fdopen(os.dup(0), "r", encoding="utf-8")
    protocol_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    devnull = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(devnull, descriptor)

    sys.dont_write_bytecode = True
    os.chdir(args.workspace)
    apply_limits(args.memory_mb)
    if not args.no_confine:
        install_confinement(args.workspace)

    namespace: dict = {"__name__": "__main__"}
    protocol_out.write(json.dumps({"ready": True}) + "\n")
    protocol_out.flush()
    for cell_number, line in enumerate(protocol_in, start=1):
        request = json.loads(line)
        set_cpu_budget(float(request.get("cpu_s", 30)))
        limit = int(request.get("max_output_chars", 20_000))
        response = execute(request["code"], namespace, cell_number, limit)
        response["id"] = request.get("id")
        protocol_out.write(json.dumps(response) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    main()
