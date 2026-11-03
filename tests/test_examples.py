import io

import pytest

from examples._common import run_example
from examples.run import NAMES, load


@pytest.mark.parametrize("name", NAMES)
def test_example_runs_offline(name, tmp_path):
    if name == "plot_chart":
        pytest.importorskip("matplotlib")
    example = load(name)
    out = io.StringIO()
    events, workspace, agent = run_example(example, out=out, workspace_root=tmp_path)
    agent.executor.close()
    final = events[-1]
    assert final.stop_reason == "end_turn"
    results = [e.result for e in events if type(e).__name__ == "ToolResultEvent"]
    assert results
    for expected in example.expected_files:
        assert (workspace / expected).stat().st_size > 0


def test_self_correct_recovers_from_error(tmp_path):
    events, _, agent = run_example(load("self_correct"), out=io.StringIO(), workspace_root=tmp_path)
    agent.executor.close()
    results = [e.result for e in events if type(e).__name__ == "ToolResultEvent"]
    assert [r.is_error for r in results] == [True, False]
    assert "KeyError: 'quantity'" in results[0].content
    assert "qty_on_hand" in results[1].content


def test_csv_analysis_numbers(tmp_path):
    events, workspace, agent = run_example(
        load("csv_analysis"), out=io.StringIO(), workspace_root=tmp_path
    )
    agent.executor.close()
    text = next(e.result.content for e in events if type(e).__name__ == "ToolResultEvent")
    import csv

    totals = {}
    for row in csv.DictReader((workspace / "sales.csv").open()):
        totals[row["region"]] = totals.get(row["region"], 0) + int(row["units"]) * float(
            row["unit_price"]
        )
    top = max(totals, key=totals.get)
    assert text.splitlines()[1].startswith(top)


def test_sample_data_is_deterministic(tmp_path):
    example = load("log_analysis")
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    example.make_data(first)
    example.make_data(second)
    assert (first / "app.log").read_text() == (second / "app.log").read_text()
