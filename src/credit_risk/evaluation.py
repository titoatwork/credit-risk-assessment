"""
Evaluation helpers for imbalanced credit risk models.

Provides multi-threshold tables, lift vs base rate, and summary metrics that
address "precision looks low" skepticism with rigorous numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def base_rate(y_true: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    return float(y.mean()) if len(y) else 0.0


def metrics_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))
    br = base_rate(y)
    lift = (prec / br) if br > 0 else float("nan")
    flagged_rate = float(pred.mean())
    return {
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "flagged_rate": flagged_rate,
        "base_rate": br,
        "lift_vs_base_rate": float(lift),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


def threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [metrics_at_threshold(y_true, proba, t) for t in thresholds]


def ranking_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "base_rate": base_rate(y),
        "brier_score": float(brier_score_loss(y, p)),
    }
    # Precision when flagging top-k% highest risk
    order = np.argsort(-p)
    n = len(y)
    for pct in (1, 5, 10, 20):
        k = max(1, int(round(n * pct / 100.0)))
        top = y[order[:k]]
        out[f"precision_at_top_{pct}pct"] = float(top.mean())
        out[f"lift_at_top_{pct}pct"] = (
            float(top.mean() / base_rate(y)) if base_rate(y) > 0 else float("nan")
        )
    return out


def full_model_report(
    name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    operating_threshold: float,
) -> dict[str, Any]:
    rank = ranking_metrics(y_true, proba)
    op = metrics_at_threshold(y_true, proba, operating_threshold)
    sweep = threshold_sweep(y_true, proba)
    return {
        "name": name,
        "ranking": rank,
        "operating_point": op,
        "threshold_sweep": sweep,
        "interpretation": {
            "note": (
                "Precision at a fixed threshold is not the primary skill metric under "
                f"class imbalance (base_rate={rank['base_rate']:.4f}). "
                "ROC-AUC/PR-AUC and lift_vs_base_rate / precision_at_top_k% show ranking skill. "
                f"At the operating threshold, lift_vs_base_rate="
                f"{op['lift_vs_base_rate']:.2f}x means flagged applicants default that many "
                "times more often than average."
            )
        },
    }
