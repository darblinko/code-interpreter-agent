"""Shared pieces for the example scripts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from code_interpreter_agent import Agent, Event, SubprocessExecutor, build_agent
from code_interpreter_agent.cli.render import render_events
from code_interpreter_agent.providers import FakeProvider, Provider
from code_interpreter_agent.providers.fake import TurnSource


@dataclass(frozen=True)
class Example:
    name: str
    prompt: str
    make_data: Callable[[Path], None]
    script: Callable[[], list[TurnSource]]  # fresh turns for each run
    expected_files: tuple[str, ...] = ()


def run_example(
    example: Example,
    provider: Provider | None = None,
    out: TextIO | None = None,
    workspace_root: Path | None = None,
) -> tuple[list[Event], Path, Agent]:
    """Run an example. Without a provider the scripted FakeProvider is used, so it needs no
    network or API key. Returns the events, the workspace path and the (still open) agent."""
    provider = provider or FakeProvider(example.script())
    executor = SubprocessExecutor(workspace_root=workspace_root, timeout_s=60)
    agent = build_agent(provider, executor=executor, max_iterations=8)
    session_id = example.name
    example.make_data(executor.workspace_path(session_id))
    events: list[Event] = []

    def capture():
        for event in agent.run(session_id, example.prompt):
            events.append(event)
            yield event

    render_events(capture(), out or sys.stdout)
    return events, executor.workspace_path(session_id), agent
