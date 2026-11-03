"""CSV analysis: revenue by region and best product, standard library only."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from code_interpreter_agent.providers.fake import TurnSource, code_turn, text_turn

from ._common import Example

REGIONS = ("north", "south", "east", "west")
PRODUCTS = {"widget": 4.5, "gadget": 12.0, "gizmo": 7.25}


def make_data(workspace: Path) -> None:
    rng = random.Random(1)
    with (workspace / "sales.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "region", "product", "units", "unit_price"])
        for day in range(200):
            product = rng.choice(list(PRODUCTS))
            writer.writerow(
                [f"2026-01-{day % 28 + 1:02d}", rng.choice(REGIONS), product,
                 rng.randint(1, 20), PRODUCTS[product]]
            )  # fmt: skip


ANALYSIS = """\
import csv
from collections import defaultdict

by_region = defaultdict(float)
by_product = defaultdict(float)
with open("sales.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        revenue = int(row["units"]) * float(row["unit_price"])
        by_region[row["region"]] += revenue
        by_product[row["product"]] += revenue

for region, total in sorted(by_region.items(), key=lambda item: -item[1]):
    print(f"{region:6} {total:10.2f}")
best = max(by_product, key=by_product.get)
print("best product:", best, round(by_product[best], 2))
"""


def script() -> list[TurnSource]:
    return [
        code_turn(ANALYSIS, "I will total revenue per region and per product."),
        text_turn("Revenue per region and the best selling product are listed above."),
    ]


EXAMPLE = Example(
    name="csv_analysis",
    prompt="sales.csv has date, region, product, units, unit_price. Which region earns the "
    "most revenue and which product sells best?",
    make_data=make_data,
    script=script,
)
