"""Command line interface: `cia` (REPL) or `cia -p "question"` (one shot)."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable, Sequence

from ..agent import Agent
from ..executors import EXECUTOR_NAMES
from ..factory import build_agent
from ..providers import PROVIDER_NAMES, MissingCredentialsError
from .render import render_events

HELP = "Commands: /reset clears the conversation and workspace, /files lists files, /exit quits."
DEFAULT_WORKSPACE_ROOT = ".cia-workspaces"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cia", description="Code interpreter agent")
    parser.add_argument("-p", "--prompt", help="run one prompt and exit")
    parser.add_argument("--provider", choices=PROVIDER_NAMES, default="anthropic")
    parser.add_argument("--model", help="model id; defaults come from the provider")
    parser.add_argument("--executor", choices=EXECUTOR_NAMES, default="subprocess")
    parser.add_argument(
        "--workspace-root",
        default=DEFAULT_WORKSPACE_ROOT,
        help="directory for session workspaces, kept after exit (default: %(default)s)",
    )
    parser.add_argument("--docker-image", help="image for the docker executor")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per code call")
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, help="total token budget per message")
    parser.add_argument("--quiet", action="store_true", help="hide tool calls and results")
    return parser.parse_args(argv)


def agent_from_args(args: argparse.Namespace) -> Agent:
    options: dict = {
        "timeout_s": args.timeout,
        "memory_mb": args.memory_mb,
        "workspace_root": args.workspace_root,
    }
    if args.executor == "docker" and args.docker_image:
        options["image"] = args.docker_image
    return build_agent(
        args.provider,
        model=args.model,
        executor=args.executor,
        executor_options=options,
        max_iterations=args.max_iterations,
        max_total_tokens=args.max_tokens,
    )


def repl(agent: Agent, session_id: str, verbose: bool) -> None:
    print(f"Session {session_id}. {HELP}")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/reset":
            agent.reset(session_id)
            print("Session reset.")
        elif line == "/files":
            workspace = agent.executor.workspace_path(session_id)
            names = sorted(p.name for p in workspace.iterdir() if not p.name.startswith("."))
            print("\n".join(names) or "(no files)")
        else:
            render_events(agent.run(session_id, line), verbose=verbose)


def main(
    argv: Sequence[str] | None = None,
    agent_factory: Callable[[argparse.Namespace], Agent] = agent_from_args,
) -> int:
    args = parse_args(argv)
    try:
        agent = agent_factory(args)
    except (MissingCredentialsError, ImportError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    try:
        if args.prompt:
            render_events(agent.run(session_id, args.prompt), verbose=not args.quiet)
        else:
            repl(agent, session_id, verbose=not args.quiet)
    except KeyboardInterrupt:
        print()
    finally:
        if agent.executor is not None:
            report_files(agent, session_id)
            agent.executor.close()
    return 0


def report_files(agent: Agent, session_id: str) -> None:
    workspace = agent.executor.workspace_path(session_id)
    if any(not p.name.startswith(".") for p in workspace.iterdir()):
        print(f"Files from this session are in {workspace}")


if __name__ == "__main__":
    raise SystemExit(main())
