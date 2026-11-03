"""Plot a chart to a file. Needs matplotlib: pip install 'code-interpreter-agent[examples]'."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from code_interpreter_agent.providers.fake import TurnSource, code_turn, text_turn

from ._common import Example


def make_data(workspace: Path) -> None:
    rng = random.Random(2)
    with (workspace / "temperatures.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "temp_c"])
        for day in range(1, 61):
            writer.writerow([day, round(12 + 8 * math.sin(day / 9) + rng.gauss(0, 1.5), 2)])


PLOT = """\
import csv
import matplotlib.pyplot as plt

days, temps = [], []
with open("temperatures.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        days.append(int(row["day"]))
        temps.append(float(row["temp_c"]))

window = 7
average = [sum(temps[max(0, i - window + 1): i + 1]) / len(temps[max(0, i - window + 1): i + 1])
           for i in range(len(temps))]
plt.figure(figsize=(8, 4))
plt.plot(days, temps, alpha=0.4, label="daily")
plt.plot(days, average, label=f"{window}-day mean")
plt.xlabel("day")
plt.ylabel("temperature (C)")
plt.legend()
plt.tight_layout()
plt.savefig("temperatures.png", dpi=120)
print("saved temperatures.png, max", max(temps))
"""


def script() -> list[TurnSource]:
    return [
        code_turn(PLOT, "I will plot the daily values and a 7 day moving average."),
        text_turn("The chart is saved as temperatures.png."),
    ]


EXAMPLE = Example(
    name="plot_chart",
    prompt="Plot temperatures.csv as a line chart with a 7 day moving average and save it as "
    "a PNG.",
    make_data=make_data,
    script=script,
    expected_files=("temperatures.png",),
)
