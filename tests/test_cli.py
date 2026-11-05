from code_interpreter_agent import FakeProvider, FinalEvent, TextDelta, build_agent
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


def test_files_are_kept_after_exit(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = [code_turn("open('parabola.png', 'wb').write(b'png')"), text_turn("saved")]

    def factory(args):
        assert args.workspace_root == ".cia-workspaces"
        return build_agent(
            FakeProvider(script), executor_options={"workspace_root": args.workspace_root}
        )

    main(["-p", "plot"], agent_factory=factory)
    saved = list((tmp_path / ".cia-workspaces").glob("cli-*/parabola.png"))
    assert len(saved) == 1 and saved[0].read_bytes() == b"png"
    assert "Files from this session are in" in capsys.readouterr().out


def test_unstreamed_final_text_is_printed(capsys):
    from code_interpreter_agent.cli.render import render_events
    from code_interpreter_agent.providers.base import Usage

    final = FinalEvent("42", "end_turn", 1, Usage())
    render_events([final], verbose=False)
    assert capsys.readouterr().out == "42\n"
    render_events([TextDelta("42"), final], verbose=False)
    assert capsys.readouterr().out == "42\n"


def test_missing_key_exits_with_error(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["-p", "hi", "--provider", "anthropic"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
