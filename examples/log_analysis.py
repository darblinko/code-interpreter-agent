"""Text analysis: summarize an application log with regular expressions."""

from __future__ import annotations

import random
from pathlib import Path

from code_interpreter_agent.providers.fake import TurnSource, code_turn, text_turn

from ._common import Example

MESSAGES = {
    "INFO": ["request served", "cache hit", "user login"],
    "WARNING": ["slow query", "retrying upstream"],
    "ERROR": ["payment timeout", "database connection refused", "payment timeout"],
}


def make_data(workspace: Path) -> None:
    rng = random.Random(3)
    lines = []
    for second in range(0, 3 * 3600, 7):
        level = rng.choices(list(MESSAGES), weights=[80, 12, 8])[0]
        stamp = f"2026-02-01 {second // 3600:02d}:{second % 3600 // 60:02d}:{second % 60:02d}"
        lines.append(f"{stamp} {level} {rng.choice(MESSAGES[level])} id={rng.randint(1, 999)}")
    (workspace / "app.log").write_text("\n".join(lines) + "\n")


ANALYSIS = """\
import re
from collections import Counter

pattern = re.compile(r"^(\\S+ \\S+) (INFO|WARNING|ERROR) (.*?) id=\\d+$")
levels, errors, errors_per_hour = Counter(), Counter(), Counter()
with open("app.log") as handle:
    for line in handle:
        match = pattern.match(line.strip())
        if not match:
            continue
        stamp, level, message = match.groups()
        levels[level] += 1
        if level == "ERROR":
            errors[message] += 1
            errors_per_hour[stamp[11:13]] += 1

print("lines by level:", dict(levels))
print("top errors:", errors.most_common(3))
print("errors per hour:", dict(sorted(errors_per_hour.items())))
"""


def script() -> list[TurnSource]:
    return [
        code_turn(ANALYSIS, "I will parse each line and count levels and error messages."),
        text_turn("The level counts, the most frequent errors and errors per hour are above."),
    ]


EXAMPLE = Example(
    name="log_analysis",
    prompt="Summarize app.log: how many lines per level, which errors are most frequent, and "
    "how do errors distribute by hour?",
    make_data=make_data,
    script=script,
)
