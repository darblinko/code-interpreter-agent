import pytest

from code_interpreter_agent import (
    Agent,
    FakeProvider,
    FinalEvent,
    PythonTool,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from code_interpreter_agent.providers import code_turn, text_turn


def run_all(agent, session, message):
    return list(agent.run(session, message))


def test_text_only_turn(executor):
    agent = Agent(FakeProvider([text_turn("hello there")]), executor)
    events = run_all(agent, "s", "hi")
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "hello there"
    final = events[-1]
    assert isinstance(final, FinalEvent) and final.stop_reason == "end_turn"
    assert final.text == "hello there" and final.iterations == 1


def test_tool_loop_event_order(executor):
    provider = FakeProvider([code_turn("print(6 * 7)", "computing"), text_turn("it is 42")])
    events = run_all(Agent(provider, executor), "s", "what is 6*7")
    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "TextDelta", "ToolCallEvent", "ToolResultEvent", *["TextDelta"] * 3, "FinalEvent",
    ]  # fmt: skip
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert "42" in result.result.content and not result.result.is_error
    assert events[-1].iterations == 2


def test_tool_error_is_fed_back_to_model(executor):
    provider = FakeProvider([code_turn("1/0"), text_turn("sorry")])
    events = run_all(Agent(provider, executor), "s", "go")
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.result.is_error and "ZeroDivisionError" in result.result.content
    second_request = provider.requests[1]
    assert second_request[-1]["content"][0]["is_error"] is True


def test_max_iterations_stops_loop(executor):
    provider = FakeProvider([code_turn(f"print({i})") for i in range(5)])
    agent = Agent(provider, executor, max_iterations=3)
    final = run_all(agent, "s", "loop")[-1]
    assert final.stop_reason == "max_iterations" and final.iterations == 3
    assert len(provider.requests) == 3


def test_token_budget_stops_loop(executor):
    provider = FakeProvider([code_turn(f"print({i})") for i in range(5)])
    agent = Agent(provider, executor, max_total_tokens=50)
    final = run_all(agent, "s", "loop")[-1]
    assert final.stop_reason == "token_budget" and final.usage.total >= 50
    assert len(provider.requests) < 5


def test_unknown_tool_is_reported_to_model(executor):
    from code_interpreter_agent import ToolCall
    from code_interpreter_agent.providers import Turn

    turn = Turn(tool_calls=(ToolCall("c1", "nope", {}),))
    events = run_all(Agent(FakeProvider([turn, text_turn("ok")]), executor), "s", "x")
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.result.is_error and "Unknown tool" in result.result.content


def test_history_persists_per_session_and_is_isolated(executor):
    provider = FakeProvider([text_turn("a1"), text_turn("b1"), text_turn("a2")])
    agent = Agent(provider, executor)
    run_all(agent, "A", "first A")
    run_all(agent, "B", "first B")
    run_all(agent, "A", "second A")
    seen_by_third_call = [m["content"] for m in provider.requests[2] if m["role"] == "user"]
    assert seen_by_third_call == ["first A", "second A"]
    assert len(agent.history("B")) == 2


def test_interpreter_state_survives_between_messages(executor):
    provider = FakeProvider(
        [code_turn("n = 5"), text_turn("done"), code_turn("print(n)"), text_turn("ok")]
    )
    agent = Agent(provider, executor)
    run_all(agent, "s", "set")
    events = run_all(agent, "s", "read")
    assert any(isinstance(e, ToolResultEvent) and "5" in e.result.content for e in events)


def test_reset_clears_history_and_executor(executor):
    agent = Agent(FakeProvider([code_turn("n = 1"), text_turn("ok")]), executor)
    run_all(agent, "s", "x")
    agent.reset("s")
    assert agent.history("s") == []
    assert not executor.run("s", "print(n)").ok


def test_default_tools_include_python(executor):
    agent = Agent(FakeProvider([]), executor)
    assert "python" in agent.tools
    assert isinstance(PythonTool(executor).input_schema, dict)


def test_custom_tool_is_dispatched(executor):
    from code_interpreter_agent import ToolCall, ToolResult
    from code_interpreter_agent.providers import Turn

    class Echo:
        name = "echo"
        description = "Echo text"
        input_schema = {"type": "object", "properties": {"text": {"type": "string"}}}

        def run(self, session_id, arguments):
            return ToolResult(f"{session_id}:{arguments['text']}")

    turn = Turn(tool_calls=(ToolCall("c1", "echo", {"text": "hey"}),))
    agent = Agent(FakeProvider([turn, text_turn("ok")]), executor, [Echo()])
    events = run_all(agent, "sess", "x")
    assert any(isinstance(e, ToolResultEvent) and e.result.content == "sess:hey" for e in events)
    assert any(isinstance(e, ToolCallEvent) for e in events)


def test_exhausted_script_raises(executor):
    with pytest.raises(RuntimeError, match="exhausted"):
        run_all(Agent(FakeProvider([]), executor), "s", "x")


def test_stopping_after_a_tool_call_keeps_history_valid(executor):
    provider = FakeProvider([code_turn("print(1)", call_id="c1"), text_turn("done")])
    agent = Agent(provider, executor)
    events = agent.run("s", "go")
    for event in events:
        if isinstance(event, ToolCallEvent):
            break
    events.close()
    last = agent.history("s")[-1]
    assert last["role"] == "user"
    assert last["content"][0]["tool_use_id"] == "c1" and last["content"][0]["is_error"]
    run_all(agent, "s", "again")
    assert provider.requests[1][2] == last


def test_provider_content_blocks_are_replayed_unchanged(executor):
    from code_interpreter_agent.providers.base import Completion, ToolCall, Usage

    blocks = (
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "tool_use", "id": "t1", "name": "python", "input": {"code": "print(5)"}},
    )
    seen = []

    class ThinkingProvider:
        name, model = "stub", "stub"

        def stream(self, *, system, messages, tools):
            seen.append([dict(m) for m in messages])
            if len(seen) == 1:
                call = ToolCall("t1", "python", {"code": "print(5)"})
                yield Completion("", (call,), Usage(1, 1), "tool_use", blocks)
            else:
                yield Completion("five", (), Usage(1, 1), "end_turn")

    final = run_all(Agent(ThinkingProvider(), executor), "s", "go")[-1]
    assert final.text == "five"
    assert seen[1][1] == {"role": "assistant", "content": list(blocks)}
    assert seen[1][2]["content"][0]["tool_use_id"] == "t1"
