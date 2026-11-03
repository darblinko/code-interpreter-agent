from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import code_interpreter_agent as package  # noqa: E402
from code_interpreter_agent.providers import code_turn, text_turn  # noqa: E402

APP = str(Path(__file__).parents[1] / "src/code_interpreter_agent/ui/app.py")


def test_ui_runs_a_turn_with_a_fake_provider(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    real_build = package.build_agent
    script = [code_turn("print(3 * 3)"), text_turn("nine")]
    monkeypatch.setattr(
        package,
        "build_agent",
        lambda *a, **k: real_build(
            package.FakeProvider(script), executor_options={"workspace_root": tmp_path}
        ),
    )
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.chat_input[0].set_value("compute").run()
    assert not app.exception
    assert [c.value for c in app.code] == ["print(3 * 3)"]
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
