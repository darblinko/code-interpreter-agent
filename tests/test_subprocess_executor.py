import time

import pytest

from code_interpreter_agent import SubprocessExecutor


def test_prints_stdout(executor):
    result = executor.run("a", "print(1 + 1)")
    assert result.ok and result.stdout == "2\n" and result.stderr == ""


def test_last_expression_is_displayed(executor):
    assert executor.run("a", "x = 21\nx * 2").stdout == "42\n"


def test_state_persists_across_calls(executor):
    executor.run("a", "values = [1, 2, 3]\ndef total():\n    return sum(values)")
    assert executor.run("a", "print(total())").stdout == "6\n"


def test_sessions_are_isolated(executor):
    executor.run("one", "secret = 'only in one'\nopen('mine.txt', 'w').write('x')")
    other = executor.run("two", "print(secret)")
    assert not other.ok and "NameError" in other.stderr
    listing = executor.run("two", "import os\nprint(os.listdir('.'))")
    assert "mine.txt" not in listing.stdout


def test_error_returns_traceback_with_source_line(executor):
    result = executor.run("a", "def f():\n    return 1 / 0\nf()")
    assert not result.ok
    assert "ZeroDivisionError" in result.stderr and "return 1 / 0" in result.stderr
    assert "worker.py" not in result.stderr
    assert executor.run("a", "print('still alive')").ok


def test_syntax_error_is_reported(executor):
    result = executor.run("a", "def broken(:")
    assert not result.ok and "SyntaxError" in result.stderr


def test_reports_created_and_modified_files(executor):
    result = executor.run("a", "open('out.csv', 'w').write('a,b')")
    assert result.files == ("out.csv",)
    assert executor.run("a", "print('no change')").files == ()
    assert executor.run("a", "open('out.csv', 'a').write('c')").files == ("out.csv",)


def test_timeout_kills_worker_and_next_call_recovers(tmp_path):
    with SubprocessExecutor(workspace_root=tmp_path, timeout_s=1, cpu_s=30) as executor:
        executor.run("a", "kept = 1")
        result = executor.run("a", "while True:\n    pass")
        assert not result.ok and "time limit" in result.stderr
        after = executor.run("a", "print('kept' in globals())")
        assert after.ok and after.stdout == "False\n"


def test_cpu_limit_terminates_worker(tmp_path):
    with SubprocessExecutor(workspace_root=tmp_path, timeout_s=20, cpu_s=1) as executor:
        result = executor.run("a", "while True:\n    pass")
        assert not result.ok and "CPU time limit" in result.stderr


def test_memory_limit_raises_memory_error(tmp_path):
    with SubprocessExecutor(workspace_root=tmp_path, memory_mb=256) as executor:
        result = executor.run("a", "block = bytearray(1024 * 1024 * 1024)")
        assert not result.ok and "MemoryError" in result.stderr
        assert executor.run("a", "print('ok')").ok


@pytest.mark.parametrize(
    "code",
    [
        "print(open('/etc/passwd').read())",
        "import pathlib; print(pathlib.Path('/etc/hostname').read_text())",
        "import os; print(os.listdir('/'))",
        "open('/tmp/escape.txt', 'w').write('x')",
        "import os; os.chdir('/'); print(open('etc/passwd').read())",
        "print(open('../other/secret.txt').read())",
    ],
)
def test_access_outside_workspace_is_refused(executor, code):
    executor.run("other", "open('secret.txt', 'w').write('s')")
    result = executor.run("a", code)
    assert not result.ok
    assert "PermissionError" in result.stderr
    assert "root:" not in result.stdout


def test_symlink_to_outside_is_refused(executor):
    result = executor.run("a", "import os; os.symlink('/etc/passwd', 'link'); open('link').read()")
    assert not result.ok and "PermissionError" in result.stderr


def test_subprocess_creation_is_refused(executor):
    for code in ("import subprocess; subprocess.run(['ls'])", "import os; os.system('ls')"):
        result = executor.run("a", code)
        assert not result.ok and "PermissionError" in result.stderr


def test_environment_does_not_inherit_secrets(executor, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    monkeypatch.setenv("SECRET_TOKEN", "should-not-leak")
    result = executor.run("a", "import os; print(sorted(os.environ))")
    assert "should-not-leak" not in result.stdout
    assert "ANTHROPIC_API_KEY" not in result.stdout and "SECRET_TOKEN" not in result.stdout


def test_working_directory_is_the_workspace(executor):
    result = executor.run("a", "import os; print(os.getcwd())")
    assert result.stdout.strip() == str(executor.workspace_path("a"))


def test_output_is_truncated(tmp_path):
    with SubprocessExecutor(workspace_root=tmp_path, max_output_chars=100) as executor:
        result = executor.run("a", "print('x' * 1000)")
        assert len(result.stdout) < 200 and "truncated" in result.stdout


def test_reset_clears_state_and_files(executor):
    executor.run("a", "kept = 1\nopen('f.txt', 'w').write('x')")
    executor.reset("a")
    assert not (executor.workspace_path("a") / "f.txt").exists()
    assert not executor.run("a", "print(kept)").ok


def test_invalid_session_id_is_rejected(executor):
    for bad in ("", "../escape", "a/b", ".hidden", "x" * 65, "a\n"):
        with pytest.raises(ValueError):
            executor.run(bad, "1")


def test_concurrent_sessions(executor):
    from concurrent.futures import ThreadPoolExecutor

    def job(name):
        executor.run(name, f"name = {name!r}")
        return executor.run(name, "print(name)").stdout.strip()

    names = [f"s{i}" for i in range(4)]
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(job, names)) == names


def test_matplotlib_chart_is_written(executor):
    pytest.importorskip("matplotlib")
    code = "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.savefig('c.png')"
    result = executor.run("a", code)
    assert result.ok, result.stderr
    assert "c.png" in result.files


def test_close_removes_owned_root():
    executor = SubprocessExecutor()
    root = executor.workspace_path("a").parent
    executor.run("a", "1")
    executor.close()
    assert not root.exists()


def test_protocol_corruption_restarts_worker(executor):
    executor.run("a", "kept = 1")
    forged = executor.run("a", "import os; os.write(4, b'not-json\\n'); print(9)")
    assert not forged.ok and "invalid protocol reply" in forged.stderr
    result = executor.run("a", "print(999)")
    assert result.ok and result.stdout == "999\n"


def test_reply_with_wrong_id_is_rejected(executor):
    forged = b'{"id": "x", "stdout": "", "stderr": "", "ok": true}\n'
    code = f"import os; os.write(4, {forged!r})"
    result = executor.run("a", code)
    assert not result.ok and "invalid protocol reply" in result.stderr
    assert executor.run("a", "print(1)").stdout == "1\n"


def test_large_output_is_bounded_in_the_worker(tmp_path):
    with SubprocessExecutor(workspace_root=tmp_path, max_output_chars=100) as executor:
        result = executor.run("a", "print('x' * 1_000_000)")
        assert result.ok
        assert "[999901 characters truncated]" in result.stdout
        assert len(result.stdout) < 200


def test_oversize_protocol_line_restarts_worker(tmp_path):
    with SubprocessExecutor(workspace_root=tmp_path, max_output_chars=100) as executor:
        result = executor.run("a", "import os; os.write(4, b'x' * 200_000)")
        assert not result.ok and "size limit" in result.stderr
        assert executor.run("a", "print(2)").stdout == "2\n"


@pytest.mark.parametrize(
    "code",
    [
        "import os, sys; os.remove(os.path.join(sys.prefix, 'pyvenv.cfg'))",
        "import os, sys; os.chmod(sys.prefix, 0o700)",
        "import os, sys; os.rename(os.path.join(sys.prefix, 'pyvenv.cfg'), 'cfg')",
        "import os, sys; os.link(os.path.join(sys.prefix, 'pyvenv.cfg'), 'cfg')",
        "import os; os.utime('/etc/passwd')",
    ],
)
def test_changes_to_the_python_installation_are_refused(executor, code):
    result = executor.run("a", code)
    assert not result.ok and "PermissionError" in result.stderr


def test_signals_and_rlimit_changes_are_refused(executor):
    kill = executor.run("a", "import os; os.kill(os.getppid(), 0)")
    assert not kill.ok and "PermissionError" in kill.stderr
    code = "import resource; resource.setrlimit(resource.RLIMIT_CPU, (10**6, 10**6))"
    limit = executor.run("a", code)
    assert not limit.ok and "PermissionError" in limit.stderr
    assert executor.run("a", "print(3)").stdout == "3\n"


def test_symlinked_session_workspace_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    (root / "s1").symlink_to(tmp_path / "outside")
    with SubprocessExecutor(workspace_root=root) as executor, pytest.raises(ValueError):
        executor.run("s1", "1")


def test_symlinks_are_not_reported_as_files(tmp_path):
    from code_interpreter_agent.executors.process import snapshot_workspace

    (tmp_path / "real.txt").write_text("x")
    (tmp_path / "leak.txt").symlink_to("/etc/passwd")
    (tmp_path / "linked_dir").symlink_to("/etc")
    assert list(snapshot_workspace(tmp_path)) == ["real.txt"]


def test_terminate_kills_the_process_group(tmp_path):
    import os
    import subprocess

    group = subprocess.Popen(["sh", "-c", "sleep 60 & wait"], start_new_session=True)
    with SubprocessExecutor(workspace_root=tmp_path) as executor:
        executor._terminate("a", group)
    group.wait()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(group.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail("a process in the worker's group survived")
