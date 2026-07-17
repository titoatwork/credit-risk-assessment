"""Imbalance-aware evaluation helpers (lift, top-k precision)."""

from __future__ import annotations

import numpy as np

from credit_risk.evaluation import (
    base_rate,
    full_model_report,
    metrics_at_threshold,
    ranking_metrics,
)


def test_base_rate_and_lift_math():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 20% base rate
    # Perfect scores: high proba for positives
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    assert abs(base_rate(y) - 0.2) < 1e-9
    m = metrics_at_threshold(y, p, 0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert abs(m["lift_vs_base_rate"] - 5.0) < 1e-9  # 1.0 / 0.2


def test_ranking_metrics_better_than_random():
    rng = np.random.RandomState(0)
    y = (rng.rand(2000) < 0.08).astype(int)
    # Score correlated with label
    p = 0.2 * y + 0.1 * rng.rand(2000)
    rank = ranking_metrics(y, p)
    assert rank["roc_auc"] > 0.6
    assert rank["pr_auc"] > rank["base_rate"]
    assert "precision_at_top_10pct" in rank


def test_full_report_structure():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.2, 0.7, 0.15, 0.3, 0.9, 0.05])
    rep = full_model_report("demo", y, p, 0.5)
    assert rep["name"] == "demo"
    assert "ranking" in rep and "operating_point" in rep
    assert len(rep["threshold_sweep"]) >= 5
