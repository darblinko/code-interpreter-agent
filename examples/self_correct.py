"""Self correction: the first attempt fails, the model reads the traceback and fixes it."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from code_interpreter_agent.providers.base import Message
from code_interpreter_agent.providers.fake import Turn, TurnSource, code_turn, text_turn

from ._common import Example


def make_data(workspace: Path) -> None:
    rng = random.Random(5)
    with (workspace / "inventory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sku", "qty_on_hand", "unit_cost"])
        for number in range(30):
            writer.writerow([f"SKU-{number:03d}", rng.randint(0, 50), round(rng.uniform(1, 40), 2)])


FIRST_ATTEMPT = """\
import csv
rows = list(csv.DictReader(open("inventory.csv")))
print(sum(int(row["quantity"]) * float(row["unit_cost"]) for row in rows))
"""

SECOND_ATTEMPT = """\
print("columns:", list(rows[0]))
print(round(sum(int(row["qty_on_hand"]) * float(row["unit_cost"]) for row in rows), 2))
"""


def fix_after_error(messages: list[Message]) -> Turn:
    """Only produce the fix if the traceback is really in the conversation."""
    last = messages[-1]["content"][0]["content"]
    if "KeyError: 'quantity'" not in last:
        raise AssertionError(f"expected a KeyError traceback, got: {last}")
    return code_turn(SECOND_ATTEMPT, "The column is called qty_on_hand. I will use that name.")


def script() -> list[TurnSource]:
    return [
        code_turn(FIRST_ATTEMPT, "I will multiply quantity by unit cost."),
        fix_after_error,
        text_turn("The total stock value is printed above."),
    ]


EXAMPLE = Example(
    name="self_correct",
    prompt="Compute the total stock value in inventory.csv (quantity times unit cost).",
    make_data=make_data,
    script=script,
)
