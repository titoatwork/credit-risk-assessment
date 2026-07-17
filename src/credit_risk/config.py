"""Project configuration and paths.

Default production artifacts point at the **full-data academic model**
(`models/full_data/`) when present; otherwise fall back to `models/`.

Override anytime with env:
  CREDIT_RISK_MODELS_DIR
  CREDIT_RISK_DATA_DIR
  CREDIT_RISK_SAMPLE_SIZE
  CREDIT_RISK_MODEL
  CREDIT_RISK_THRESHOLD
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: credit-risk-assessment/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default Home Credit dataset location (user-provided)
DEFAULT_DATA_DIR = Path(
    r"C:\Users\Ibteshamul Haque\Downloads\home-credit-default-risk"
)

# Allow override via env
DATA_DIR = Path(os.environ.get("CREDIT_RISK_DATA_DIR", str(DEFAULT_DATA_DIR)))
APPLICATION_TRAIN = DATA_DIR / "application_train.csv"
COLUMNS_DESCRIPTION = DATA_DIR / "HomeCredit_columns_description.csv"

# Models: prefer full-data pack if available and no explicit override
_FULL_DATA_DIR = PROJECT_ROOT / "models" / "full_data"
_DEFAULT_MODELS = (
    _FULL_DATA_DIR
    if (_FULL_DATA_DIR / "credit_risk_bundle.joblib").exists()
    else (PROJECT_ROOT / "models")
)
MODELS_DIR = Path(os.environ.get("CREDIT_RISK_MODELS_DIR", str(_DEFAULT_MODELS)))
ARTIFACTS_PATH = MODELS_DIR / "credit_risk_bundle.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Sample-model pack (50k) kept for comparison — not the default serve path
SAMPLE_MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_ARTIFACTS_PATH = SAMPLE_MODELS_DIR / "credit_risk_bundle.joblib"
SAMPLE_METRICS_PATH = SAMPLE_MODELS_DIR / "metrics.json"
FULL_MODELS_DIR = _FULL_DATA_DIR
FULL_ARTIFACTS_PATH = _FULL_DATA_DIR / "credit_risk_bundle.joblib"
FULL_METRICS_PATH = _FULL_DATA_DIR / "metrics.json"

# Model / decision settings
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
# Fallback only when threshold not stored in the bundle
DEFAULT_THRESHOLD = float(os.environ.get("CREDIT_RISK_THRESHOLD", "0.5"))
RANDOM_STATE = 42
TEST_SIZE = 0.2
# 0 = full dataset for CLI default academic runs; override with CREDIT_RISK_SAMPLE_SIZE
DEFAULT_SAMPLE_SIZE = int(os.environ.get("CREDIT_RISK_SAMPLE_SIZE", "0"))

# Production model preference when training with --model
PREFERRED_MODEL = os.environ.get("CREDIT_RISK_MODEL", "auto")

# SHAP
SHAP_TOP_K = 8

# Project status (documentation sync)
PROJECT_STATUS = {
    "primary_model_pack": "models/full_data",
    "primary_rows": 307511,
    "primary_production_model": "xgboost",
    "primary_test_roc_auc": 0.769,
    "lifecycle": "academic_v2_train_val_test",
    "hyperparameter_tuning_on_full_data": True,
    "sample_model_pack": "models/ (50k comparison)",
}
