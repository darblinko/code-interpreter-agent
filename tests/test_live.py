"""Opt-in tests against a real model. Run: CIA_LIVE_TESTS=1 ANTHROPIC_API_KEY=... pytest -m live"""

import os

import pytest

from code_interpreter_agent import build_agent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("CIA_LIVE_TESTS") != "1", reason="set CIA_LIVE_TESTS=1"),
]


@pytest.mark.parametrize(
    "provider,key", [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")]
)
def test_live_arithmetic(provider, key, tmp_path):
    if not os.environ.get(key):
        pytest.skip(f"{key} not set")
    agent = build_agent(provider, executor_options={"workspace_root": tmp_path})
    try:
        events = list(
            agent.run("live", "Use python to compute 123456 * 654321 and state the result.")
        )
    finally:
        agent.executor.close()
    assert "80779853376" in events[-1].text.replace(",", "")
