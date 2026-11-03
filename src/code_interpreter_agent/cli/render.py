"""Plain-text rendering of agent events for the terminal."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO

from ..agent import Event, FinalEvent, TextDelta, ToolCallEvent, ToolResultEvent


def render_events(events: Iterable[Event], out: TextIO = sys.stdout, verbose: bool = True) -> str:
    """Print events as they arrive and return the final answer text."""
    final_text = ""
    streamed = False
    for event in events:
        if isinstance(event, TextDelta):
            out.write(event.text)
            out.flush()
            streamed = True
        elif isinstance(event, ToolCallEvent) and verbose:
            code = event.call.arguments.get("code", event.call.arguments)
            out.write(f"\n[tool call: {event.call.name}]\n{code}\n")
        elif isinstance(event, ToolResultEvent) and verbose:
            label = "error" if event.result.is_error else "result"
            out.write(f"[tool {label}]\n{event.result.content}\n")
        elif isinstance(event, FinalEvent):
            final_text = event.text
            if event.stop_reason not in ("end_turn",):
                out.write(f"\n[stopped: {event.stop_reason} after {event.iterations} steps]")
            if verbose:
                out.write(f"\n[tokens: {event.usage.input_tokens} in, {event.usage.output_tokens} out]")
    if streamed or final_text:
        out.write("\n")
    return final_text
