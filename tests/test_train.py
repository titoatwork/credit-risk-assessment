"""Training path uses real data and reports real ROC-AUC for both models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_risk.config import APPLICATION_TRAIN
from credit_risk.train import train_models


@pytest.mark.skipif(not APPLICATION_TRAIN.exists(), reason="Home Credit data missing")
def test_train_academic_lifecycle_reports_roc_auc(tmp_path: Path):
    artifacts = tmp_path / "bundle.joblib"
    metrics_path = tmp_path / "metrics.json"
    result = train_models(
        sample_size=4000,
        artifacts_path=artifacts,
        metrics_path=metrics_path,
        preferred_model="auto",
        run_eda=False,
    )
    assert artifacts.exists()
    assert metrics_path.exists()

    names = {m.name for m in result.metrics}
    assert "logistic_regression" in names
    assert "xgboost" in names
    assert "dummy_stratified" in names

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["lifecycle"] == "academic_v2_train_val_test"
    assert payload["split"]["threshold_selected_on"] == "validation"
    assert payload["split"]["final_metrics_on"] == "test"
    assert payload["split"]["n_validation"] > 0

    dummy_auc = payload["models"]["dummy_stratified"]["roc_auc"]
    lr_auc = payload["models"]["logistic_regression"]["roc_auc"]
    xgb_auc = payload["models"]["xgboost"]["roc_auc"]
    # Dummy ~0.5; real models must beat it meaningfully on real data
    assert 0.45 <= dummy_auc <= 0.60
    assert lr_auc > dummy_auc + 0.05
    assert xgb_auc > dummy_auc + 0.05
    assert result.production_model in {"xgboost", "logistic_regression"}
    assert payload["threshold_info"].get("selected_on") in {"validation", "manual"}
