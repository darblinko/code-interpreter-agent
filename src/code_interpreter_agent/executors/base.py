"""Executor protocol and result type."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    ok: bool
    files: tuple[str, ...] = field(default_factory=tuple)
    duration_s: float = 0.0


class Executor(Protocol):
    def run(self, session_id: str, code: str) -> ExecResult:
        """Run code in the session's persistent interpreter. Variables survive between calls."""
        ...

    def reset(self, session_id: str) -> None:
        """Discard the session's interpreter state and workspace files."""
        ...

    def close(self) -> None:
        """Stop every session and release resources."""
        ...

    def workspace_path(self, session_id: str) -> Path:
        """Directory that holds the session's files. Created on first use."""
        ...


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(
            f"Invalid session id {session_id!r}: use 1 to 64 characters from A-Z a-z 0-9 _ . -, "
            "starting with a letter or digit"
        )
    return session_id


def read_workspace_file(workspace: Path, name: str, max_bytes: int) -> bytes:
    """Read a regular file inside the workspace without following symbolic links in any path
    component. Executed code controls the workspace, so a reported file name may point at a
    link to a host file; this refuses it. Raises OSError or ValueError on refusal."""
    parts = PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
        raise ValueError(f"Not a workspace-relative file name: {name!r}")
    directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Not a regular file: {name!r}")
        if info.st_size > max_bytes:
            raise ValueError(f"{name!r} is larger than {max_bytes} bytes")
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{name!r} is larger than {max_bytes} bytes")
    return data
