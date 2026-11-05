"""Plain-text rendering of agent events for the terminal."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO

from ..agent import Event, FinalEvent, TextDelta, ToolCallEvent, ToolResultEvent

# Stop reasons whose FinalEvent text comes from the last model call, not an earlier one.
CURRENT_TURN = ("end_turn", "max_tokens")


def render_events(events: Iterable[Event], out: TextIO | None = None, verbose: bool = True) -> str:
    """Print events as they arrive and return the final answer text."""
    out = out or sys.stdout
    final_text = ""
    streamed = False
    streamed_this_turn = False
    for event in events:
        if isinstance(event, TextDelta):
            out.write(event.text)
            out.flush()
            streamed = streamed_this_turn = True
        elif isinstance(event, ToolCallEvent):
            streamed_this_turn = False
            if verbose:
                code = event.call.arguments.get("code", event.call.arguments)
                out.write(f"\n[tool call: {event.call.name}]\n{code}\n")
        elif isinstance(event, ToolResultEvent) and verbose:
            label = "error" if event.result.is_error else "result"
            out.write(f"[tool {label}]\n{event.result.content}\n")
        elif isinstance(event, FinalEvent):
            final_text = event.text
            if final_text and not streamed_this_turn and event.stop_reason in CURRENT_TURN:
                out.write(final_text)
            if event.stop_reason not in ("end_turn",):
                out.write(f"\n[stopped: {event.stop_reason} after {event.iterations} steps]")
            if verbose:
                out.write(
                    f"\n[tokens: {event.usage.input_tokens} in, {event.usage.output_tokens} out]"
                )
    if streamed or final_text:
        out.write("\n")
    return final_text
