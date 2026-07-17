"""Shared fixtures: train a small real model once per session for tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk.config import APPLICATION_TRAIN  # noqa: E402
from credit_risk.train import train_models  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def scratch_models_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("credit_risk_models")


@pytest.fixture(scope="session")
def trained_bundle(scratch_models_dir: Path):
    """
    Train LR + XGBoost on a small stratified sample of the REAL Home Credit
    application_train.csv (same code path as production training).
    """
    if not APPLICATION_TRAIN.exists():
        pytest.skip(f"Dataset missing: {APPLICATION_TRAIN}")

    artifacts = scratch_models_dir / "credit_risk_bundle.joblib"
    metrics = scratch_models_dir / "metrics.json"
    # Keep small for CI/laptop stability after prior crash risk
    result = train_models(
        sample_size=4000,
        artifacts_path=artifacts,
        metrics_path=metrics,
        preferred_model="xgboost",
        run_eda=False,
    )
    assert artifacts.exists()
    assert metrics.exists()
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert "logistic_regression" in payload["models"]
    assert "xgboost" in payload["models"]
    return {
        "artifacts_path": artifacts,
        "metrics_path": metrics,
        "result": result,
        "metrics": payload,
    }


@pytest.fixture
def bundle(trained_bundle):
    from credit_risk.predict import load_bundle, reset_bundle_cache

    reset_bundle_cache()
    b = load_bundle(path=trained_bundle["artifacts_path"], force_reload=True)
    yield b
    reset_bundle_cache()


@pytest.fixture
def high_risk_features(bundle) -> dict:
    """
    Build a feature dict biased toward high default risk using real column names.
    EXT_SOURCE_* are strong Home Credit predictors; low values => higher risk.
    """
    feats = {c: None for c in bundle.feature_columns}
    # Populate known risk-driving numeric fields when present
    overrides = {
        "AMT_INCOME_TOTAL": 40000.0,
        "AMT_CREDIT": 900000.0,
        "AMT_ANNUITY": 50000.0,
        "AMT_GOODS_PRICE": 900000.0,
        "DAYS_BIRTH": -8000,  # younger
        "DAYS_EMPLOYED": -100,
        "CNT_CHILDREN": 3,
        "EXT_SOURCE_1": 0.05,
        "EXT_SOURCE_2": 0.05,
        "EXT_SOURCE_3": 0.05,
        "DAYS_ID_PUBLISH": -100,
        "DAYS_REGISTRATION": -200,
        "OWN_CAR_AGE": 20,
        "CNT_FAM_MEMBERS": 5,
        "REGION_RATING_CLIENT": 3,
        "REGION_RATING_CLIENT_W_CITY": 3,
        "HOUR_APPR_PROCESS_START": 3,
        "REG_REGION_NOT_LIVE_REGION": 1,
        "REG_CITY_NOT_LIVE_CITY": 1,
        "FLAG_DOCUMENT_3": 0,
    }
    for k, v in overrides.items():
        if k in feats:
            feats[k] = v
    cat_defaults = {
        "CODE_GENDER": "M",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "N",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Lower secondary",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "With parents",
        "OCCUPATION_TYPE": "Laborers",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
    }
    for k, v in cat_defaults.items():
        if k in feats:
            feats[k] = v
    return {k: v for k, v in feats.items() if v is not None}


@pytest.fixture
def low_risk_features(bundle) -> dict:
    feats = {}
    overrides = {
        "AMT_INCOME_TOTAL": 300000.0,
        "AMT_CREDIT": 200000.0,
        "AMT_ANNUITY": 15000.0,
        "AMT_GOODS_PRICE": 200000.0,
        "DAYS_BIRTH": -20000,
        "DAYS_EMPLOYED": -4000,
        "CNT_CHILDREN": 0,
        "EXT_SOURCE_1": 0.85,
        "EXT_SOURCE_2": 0.85,
        "EXT_SOURCE_3": 0.85,
        "REGION_RATING_CLIENT": 1,
        "REGION_RATING_CLIENT_W_CITY": 1,
    }
    for k, v in overrides.items():
        if k in bundle.feature_columns:
            feats[k] = v
    cat_defaults = {
        "CODE_GENDER": "F",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Managers",
        "ORGANIZATION_TYPE": "School",
    }
    for k, v in cat_defaults.items():
        if k in bundle.feature_columns:
            feats[k] = v
    return feats
