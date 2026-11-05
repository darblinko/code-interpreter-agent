"""Argument construction is tested offline. The integration test needs Docker and an image
with Python 3.11+, and runs only with CIA_DOCKER_TESTS=1."""

import os
import shutil
from pathlib import Path

import pytest

from code_interpreter_agent import DockerExecutor
from code_interpreter_agent.executors.docker_executor import CONTAINER_WORKER


def pair_after(command, flag):
    return command[command.index(flag) + 1]


@pytest.fixture
def command(tmp_path):
    executor = DockerExecutor(workspace_root=tmp_path, memory_mb=512)
    return executor.build_run_command("cia-test", Path(tmp_path / "s1"))


def test_network_is_off_by_default(command):
    assert pair_after(command, "--network") == "none"


def test_network_can_be_enabled(tmp_path):
    executor = DockerExecutor(workspace_root=tmp_path, network="bridge")
    assert pair_after(executor.build_run_command("n", tmp_path), "--network") == "bridge"


def test_filesystem_and_privilege_flags(command):
    assert "--read-only" in command
    assert pair_after(command, "--cap-drop") == "ALL"
    assert pair_after(command, "--security-opt") == "no-new-privileges"
    assert pair_after(command, "--user") == f"{os.getuid()}:{os.getgid()}"


def test_resource_limits(command):
    assert pair_after(command, "--memory") == "512m"
    assert pair_after(command, "--memory-swap") == "512m"
    assert pair_after(command, "--cpus") == "1.0"
    assert pair_after(command, "--pids-limit") == "128"


def test_only_workspace_is_writable(command):
    volumes = [command[i + 1] for i, part in enumerate(command) if part == "--volume"]
    writable = [v for v in volumes if v.endswith(":rw")]
    assert len(writable) == 1 and writable[0].endswith(":/workspace:rw")
    assert any(v.endswith(f"{CONTAINER_WORKER}:ro") for v in volumes)


def test_image_and_worker_command_come_last(command):
    image_index = command.index("python:3.12-slim")
    assert command[image_index + 1 : image_index + 4] == ["python", "-I", CONTAINER_WORKER]
    assert "--no-confine" in command


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("CIA_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="set CIA_DOCKER_TESTS=1 with a Docker daemon to run",
)
def test_docker_roundtrip(tmp_path):
    image = os.environ.get("CIA_DOCKER_IMAGE", "python:3.12-slim")
    with DockerExecutor(workspace_root=tmp_path, image=image, timeout_s=30) as executor:
        executor.run("d1", "value = 7\nopen('f.txt', 'w').write('x')")
        result = executor.run("d1", "print(value)")
        assert result.ok and result.stdout == "7\n"
        assert (executor.workspace_path("d1") / "f.txt").exists()
        interfaces = executor.run("d1", "import os; print(sorted(os.listdir('/sys/class/net')))")
        assert interfaces.stdout == "['lo']\n"
        root = executor.run(
            "d1", "print([l.split()[3] for l in open('/proc/mounts') if l.split()[1] == '/'])"
        )
        assert root.stdout.startswith("['ro,") or root.stdout.startswith("['ro']")
        link = executor.run("d1", "import os; os.symlink('/etc/hostname', 'leak.txt')")
        assert link.ok and link.files == ()


def test_startup_failure_reports_docker_stderr(tmp_path):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\necho 'Unable to find image nope:latest' >&2\nexit 125\n")
    fake_docker.chmod(0o755)
    with DockerExecutor(workspace_root=tmp_path / "ws", docker_binary=str(fake_docker)) as executor:
        result = executor.run("d1", "1")
    assert not result.ok
    assert "Worker failed to start. Unable to find image nope:latest" in result.stderr
