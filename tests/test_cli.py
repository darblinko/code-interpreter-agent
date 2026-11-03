from code_interpreter_agent import FakeProvider, build_agent
from code_interpreter_agent.cli.main import main
from code_interpreter_agent.providers import code_turn, text_turn


def factory_with(script, tmp_path):
    return lambda args: build_agent(
        FakeProvider(script), executor_options={"workspace_root": tmp_path}
    )


def test_one_shot_prompt(capsys, tmp_path):
    script = [code_turn("print(2 + 2)"), text_turn("four")]
    code = main(["-p", "add"], agent_factory=factory_with(script, tmp_path))
    out = capsys.readouterr().out
    assert code == 0 and "[tool call: python]" in out and "4" in out and "four" in out


def test_quiet_hides_tool_output(capsys, tmp_path):
    script = [code_turn("print(2 + 2)"), text_turn("four")]
    main(["-p", "add", "--quiet"], agent_factory=factory_with(script, tmp_path))
    assert "[tool call" not in capsys.readouterr().out


def test_repl_commands(monkeypatch, capsys, tmp_path):
    script = [code_turn("open('a.txt', 'w').write('x')"), text_turn("done")]
    lines = iter(["write a file", "/files", "/reset", "/files", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    main([], agent_factory=factory_with(script, tmp_path))
    out = capsys.readouterr().out
    assert "a.txt" in out and "Session reset." in out and "(no files)" in out


def test_missing_key_exits_with_error(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["-p", "hi", "--provider", "anthropic"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
