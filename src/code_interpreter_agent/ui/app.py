"""Streamlit demo UI. Run: uv run --extra ui streamlit run src/code_interpreter_agent/ui/app.py

Reads provider keys from the environment like every other entry point."""

from __future__ import annotations

import uuid

import streamlit as st

from code_interpreter_agent import (
    Agent,
    FinalEvent,
    MissingCredentialsError,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    build_agent,
)
from code_interpreter_agent.executors import EXECUTOR_NAMES, read_workspace_file
from code_interpreter_agent.providers import PROVIDER_NAMES

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg")
MAX_FILE_BYTES = 25_000_000


def make_agent(provider: str, model: str, executor: str) -> Agent | None:
    try:
        return build_agent(provider, model=model or None, executor=executor)
    except (MissingCredentialsError, ImportError) as error:
        st.error(str(error))
        return None


def show_files(agent: Agent, session_id: str, names: tuple[str, ...]) -> None:
    workspace = agent.executor.workspace_path(session_id)
    for name in names:
        try:
            data = read_workspace_file(workspace, name, MAX_FILE_BYTES)
        except (OSError, ValueError) as error:
            st.caption(f"{name} not shown: {error}")
            continue
        if name.lower().endswith(IMAGE_SUFFIXES):
            st.image(data, caption=name)
        else:
            st.download_button(
                f"Download {name}",
                data,
                file_name=name.rsplit("/", 1)[-1],
                key=f"dl-{uuid.uuid4()}",
            )


def close_agent(agent: Agent | None) -> None:
    if agent is not None and agent.executor is not None:
        agent.executor.close()


def stream_reply(agent: Agent, session_id: str, prompt: str) -> str:
    """Render one turn live and return the final answer text."""
    text_box = st.empty()
    text = ""
    answer = ""
    for event in agent.run(session_id, prompt):
        if isinstance(event, TextDelta):
            text += event.text
            text_box.markdown(text)
        elif isinstance(event, ToolCallEvent):
            text, text_box = "", st.empty()
            with st.expander(f"Code ({event.call.name})", expanded=True):
                st.code(event.call.arguments.get("code", str(event.call.arguments)), "python")
        elif isinstance(event, ToolResultEvent):
            with st.expander("Result", expanded=True):
                (st.error if event.result.is_error else st.text)(event.result.content)
                show_files(agent, session_id, event.result.files)
        elif isinstance(event, FinalEvent):
            answer = event.text
            if answer and not text and event.stop_reason in ("end_turn", "max_tokens"):
                text_box.markdown(answer)
            if event.stop_reason != "end_turn":
                st.warning(f"Stopped: {event.stop_reason} after {event.iterations} steps")
    return answer


def main() -> None:
    st.set_page_config(page_title="Code interpreter agent")
    st.title("Code interpreter agent")
    with st.sidebar:
        provider = st.selectbox("Provider", PROVIDER_NAMES)
        model = st.text_input("Model (blank for default)")
        executor = st.selectbox("Executor", EXECUTOR_NAMES)
        if st.button("Apply and reset session"):
            close_agent(st.session_state.get("agent"))
            st.session_state.clear()
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.agent = make_agent(provider, model, executor)
        st.session_state.transcript = []
    agent = st.session_state.agent
    if agent is None:
        st.stop()
    for role, text in st.session_state.transcript:
        st.chat_message(role).write(text)
    if prompt := st.chat_input("Ask for an analysis"):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            answer = stream_reply(agent, st.session_state.session_id, prompt)
        st.session_state.transcript += [("user", prompt), ("assistant", answer)]


main()
