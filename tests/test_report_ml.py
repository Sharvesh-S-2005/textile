"""Tests for ml/report_ml.py (consolidated ML documents from saved metrics)."""

from pathlib import Path

import pytest

from ml.report_ml import NOT_ATTEMPTED, comparison_rows, deep_learning_text, load_metrics, write_results_ml

METRICS = load_metrics()
HAVE_ALL = all(METRICS[k] is not None for k in ("forecast", "classification", "ablation", "anomaly"))
needs_metrics = pytest.mark.skipif(not HAVE_ALL, reason="ML metrics not generated in this checkout")


@needs_metrics
def test_every_group_lists_its_baseline_beside_the_models() -> None:
    rows = comparison_rows(METRICS)
    for task in ("A. Forecast", "B. Exceedance"):
        groups = {(r["target"], r["h"]) for r in rows if r["task"] == task}
        for target, h in groups:
            models = [r["model"] for r in rows if r["task"] == task and r["target"] == target and r["h"] == h]
            assert models[0] == "persistence"                    # baseline listed first


@needs_metrics
def test_classification_rows_carry_the_prevalence() -> None:
    rows = [r for r in comparison_rows(METRICS) if r["task"] == "B. Exceedance"]
    assert rows and all("prev." in r["target"] for r in rows)
    assert all(r["metric"] == "PR-AUC" for r in rows)            # PR-AUC leads, not accuracy


@needs_metrics
def test_results_ml_fills_both_markers(tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    body.write_text("## Table\n{{COMPARISON_TABLE}}\n## DL\n{{DEEP_LEARNING}}\n", encoding="utf-8")
    out = tmp_path / "out.md"
    write_results_ml(METRICS, body, out)
    text = out.read_text(encoding="utf-8")
    assert "{{" not in text and "SIMULATED DATA" in text and "| persistence |" in text


@needs_metrics
def test_missing_marker_fails_loudly(tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    body.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_results_ml(METRICS, body, tmp_path / "out.md")


def test_deep_learning_text_states_the_status() -> None:
    assert deep_learning_text(None) == NOT_ATTEMPTED
    abandoned = deep_learning_text({"status": "abandoned", "reason": "no useful training within 45 min"})
    assert abandoned.startswith("Attempted and abandoned") and "45 min" in abandoned
