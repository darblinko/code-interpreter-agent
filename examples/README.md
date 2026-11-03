# Examples

Each example generates its own sample data in the session workspace, then runs the agent.
By default a scripted FakeProvider replays recorded model turns, so no API key or network is
needed. The code in each script is executed for real by the subprocess executor.

    uv run python -m examples.run --list
    uv run python -m examples.run csv_analysis
    uv run python -m examples.run csv_analysis --live anthropic   # needs ANTHROPIC_API_KEY

Output files are written to `workspaces/<example>/`. Each example has an offline test in
`tests/test_examples.py`.

| Name | Data generated | What it shows |
|---|---|---|
| csv_analysis | sales.csv, 200 rows | Aggregation with csv and collections, no extra packages |
| plot_chart | temperatures.csv, 60 rows | A matplotlib chart saved as PNG (needs matplotlib) |
| log_analysis | app.log, about 1500 lines | Regular expressions and counters over text |
| math_stats | measurements.json, 2 x 40 values | Welch t statistic, confidence interval, p value |
| self_correct | inventory.csv, 30 rows | First attempt raises KeyError, second attempt reads the traceback and fixes the column name |

With `--live`, the real model chooses its own code, so output differs from the scripted run.
