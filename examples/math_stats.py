"""Statistics problem: compare two samples with a Welch t statistic. The p value and the
interval use the normal approximation, which the standard library supports; the exact Welch
t distribution gives p = 0.0005 for this data."""

from __future__ import annotations

import json
import random
from pathlib import Path

from code_interpreter_agent.providers.fake import TurnSource, code_turn, text_turn

from ._common import Example


def make_data(workspace: Path) -> None:
    rng = random.Random(4)
    samples = {
        "control": [round(rng.gauss(50.0, 6.0), 2) for _ in range(40)],
        "treatment": [round(rng.gauss(54.0, 6.0), 2) for _ in range(40)],
    }
    (workspace / "measurements.json").write_text(json.dumps(samples))


ANALYSIS = """\
import json
import math
import statistics
from statistics import NormalDist

samples = json.load(open("measurements.json"))
a, b = samples["control"], samples["treatment"]
mean_a, mean_b = statistics.mean(a), statistics.mean(b)
var_a, var_b = statistics.variance(a), statistics.variance(b)
standard_error = math.sqrt(var_a / len(a) + var_b / len(b))
t_statistic = (mean_b - mean_a) / standard_error
# Normal approximation to the t distribution; adequate for 40 samples per group.
p_value = 2 * (1 - NormalDist().cdf(abs(t_statistic)))
margin = 1.96 * standard_error
print(f"means: control {mean_a:.2f}, treatment {mean_b:.2f}")
print(f"difference {mean_b - mean_a:.2f}, approximate 95% CI ({mean_b - mean_a - margin:.2f}, "
      f"{mean_b - mean_a + margin:.2f})")
print(f"Welch t = {t_statistic:.2f}, p = {p_value:.4f} (normal approximation)")
"""


def script() -> list[TurnSource]:
    return [
        code_turn(ANALYSIS, "I will compute the Welch t statistic and a confidence interval."),
        text_turn(
            "The treatment mean, the confidence interval and the p value are above. The p value "
            "and interval use the normal approximation to the t distribution."
        ),
    ]


EXAMPLE = Example(
    name="math_stats",
    prompt="measurements.json has two samples, control and treatment. Is the treatment mean "
    "different? Give the difference, a 95% confidence interval and a p value.",
    make_data=make_data,
    script=script,
)
