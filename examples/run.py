"""Run an example. Default is the scripted FakeProvider (offline).

python -m examples.run --list
python -m examples.run csv_analysis
python -m examples.run csv_analysis --live anthropic
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from code_interpreter_agent.providers import get_provider

NAMES = ("csv_analysis", "plot_chart", "log_analysis", "math_stats", "self_correct")


def load(name: str):
    return importlib.import_module(f"examples.{name}").EXAMPLE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", choices=NAMES)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--live", choices=("anthropic", "openai"), help="use a real model")
    args = parser.parse_args()
    if args.list or not args.name:
        print("\n".join(NAMES))
        return
    from ._common import run_example

    example = load(args.name)
    provider = get_provider(args.live) if args.live else None
    _, workspace, agent = run_example(example, provider, workspace_root=Path("workspaces"))
    agent.executor.close()
    print(f"\nWorkspace: {workspace}")


if __name__ == "__main__":
    main()
