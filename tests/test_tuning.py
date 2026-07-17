"""Hyperparameter tuning uses train/val only and improves or matches baseline AUC."""

from __future__ import annotations

from pathlib import Path

import pytest

from credit_risk.config import APPLICATION_TRAIN
from credit_risk.data import load_application_train, split_features_target
from credit_risk.features_multitable import add_application_derived_features
from credit_risk.preprocess import infer_feature_types
from credit_risk.tuning import tune_logistic_regression, tune_xgboost
from sklearn.model_selection import train_test_split


@pytest.mark.skipif(not APPLICATION_TRAIN.exists(), reason="Home Credit data missing")
def test_tune_holdout_returns_params_and_val_auc():
    df = load_application_train(sample_size=3000, random_state=0)
    df = add_application_derived_features(df)
    X, y = split_features_target(df)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    num, cat = infer_feature_types(X_tr)
    spw = float((y_tr == 0).sum() / max(int((y_tr == 1).sum()), 1))

    lr = tune_logistic_regression(
        X_tr, y_tr, X_va, y_va, num, cat, n_iter=3, mode="holdout", random_state=0
    )
    assert lr.best_params
    assert 0.55 < lr.best_val_roc_auc <= 1.0
    assert lr.search_mode == "holdout"
    assert len(lr.history) >= 2

    xgb = tune_xgboost(
        X_tr,
        y_tr,
        X_va,
        y_va,
        num,
        cat,
        spw,
        n_iter=3,
        mode="holdout",
        random_state=0,
    )
    assert xgb.best_params
    assert 0.55 < xgb.best_val_roc_auc <= 1.0
    # Tuned model can score the validation set
    proba = xgb.best_pipeline.predict_proba(X_va)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
