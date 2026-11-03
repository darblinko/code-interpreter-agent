import pytest

from code_interpreter_agent import SubprocessExecutor


@pytest.fixture
def executor(tmp_path):
    instance = SubprocessExecutor(workspace_root=tmp_path / "ws", timeout_s=20, memory_mb=1024)
    yield instance
    instance.close()
