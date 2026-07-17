"""
Hyperparameter tuning for academic train/val/test protocol.

Search fits candidates on TRAIN and scores ROC-AUC on VALIDATION only.
The TEST set is never used for tuning.

Two modes:
  - holdout (default): random configs scored on the validation split (fast, clear)
  - cv: RandomizedSearchCV with StratifiedKFold on the train set only
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_risk.config import RANDOM_STATE
from credit_risk.preprocess import build_preprocessor

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    best_pipeline: Pipeline
    best_params: dict[str, Any]
    best_val_roc_auc: float
    best_val_pr_auc: float
    search_mode: str
    n_candidates: int
    history: list[dict[str, Any]]
    elapsed_seconds: float


def _lr_param_space(rng: np.random.RandomState, n: int) -> list[dict[str, Any]]:
    Cs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    # lbfgs only — saga is slow and often fails to converge on this feature scale
    solvers = ["lbfgs"]
    out = []
    for _ in range(n):
        out.append(
            {
                "classifier__C": float(rng.choice(Cs)),
                "classifier__solver": str(rng.choice(solvers)),
                "classifier__max_iter": int(rng.choice([1000, 2000])),
            }
        )
    # Always include a strong default
    out.append({"classifier__C": 1.0, "classifier__solver": "lbfgs", "classifier__max_iter": 1000})
    # Dedupe
    uniq = []
    seen = set()
    for p in out:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq[: max(n, 1)]


def _xgb_param_space(
    rng: np.random.RandomState,
    n: int,
    scale_pos_weight: float,
) -> list[dict[str, Any]]:
    out = []
    for _ in range(n):
        out.append(
            {
                "classifier__n_estimators": int(rng.choice([150, 250, 350, 450])),
                "classifier__max_depth": int(rng.choice([3, 4, 5, 6, 7])),
                "classifier__learning_rate": float(rng.choice([0.03, 0.05, 0.08, 0.1])),
                "classifier__min_child_weight": int(rng.choice([1, 3, 5, 10])),
                "classifier__subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                "classifier__colsample_bytree": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                "classifier__reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0])),
                "classifier__gamma": float(rng.choice([0.0, 0.1, 0.5])),
                "classifier__scale_pos_weight": float(
                    rng.choice([scale_pos_weight * 0.5, scale_pos_weight, scale_pos_weight * 1.5])
                ),
            }
        )
    # Default strong config
    out.append(
        {
            "classifier__n_estimators": 400,
            "classifier__max_depth": 6,
            "classifier__learning_rate": 0.05,
            "classifier__min_child_weight": 3,
            "classifier__subsample": 0.85,
            "classifier__colsample_bytree": 0.85,
            "classifier__reg_lambda": 1.0,
            "classifier__gamma": 0.0,
            "classifier__scale_pos_weight": scale_pos_weight,
        }
    )
    uniq = []
    seen = set()
    for p in out:
        key = tuple(sorted((k, round(v, 6) if isinstance(v, float) else v) for k, v in p.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq[: max(n, 1)]


def _base_lr_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", pre), ("classifier", clf)])


def _base_xgb_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_pos_weight: float,
) -> Pipeline:
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("preprocessor", pre), ("classifier", clf)])


def _score_val(
    pipe: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, float]:
    proba = pipe.predict_proba(X_val)[:, 1]
    return (
        float(roc_auc_score(y_val, proba)),
        float(average_precision_score(y_val, proba)),
    )


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_iter: int = 8,
    mode: str = "holdout",
    random_state: int = RANDOM_STATE,
) -> TuneResult:
    """Tune LR hyperparameters; TEST must not be passed in."""
    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    history: list[dict[str, Any]] = []

    if mode == "cv":
        pipe = _base_lr_pipeline(numeric_cols, categorical_cols)
        param_dist = {
            "classifier__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "classifier__solver": ["lbfgs"],
            "classifier__max_iter": [1000, 2000],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        # Validate on holdout val for reporting consistency
        val_auc, val_pr = _score_val(best, X_val, y_val)
        for i, params in enumerate(search.cv_results_["params"]):
            history.append(
                {
                    "params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in params.items()},
                    "mean_cv_roc_auc": float(search.cv_results_["mean_test_score"][i]),
                }
            )
        return TuneResult(
            best_pipeline=best,
            best_params={k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in search.best_params_.items()},
            best_val_roc_auc=val_auc,
            best_val_pr_auc=val_pr,
            search_mode="cv",
            n_candidates=n_iter,
            history=history,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # holdout random search
    candidates = _lr_param_space(rng, n_iter)
    best_pipe: Optional[Pipeline] = None
    best_params: dict[str, Any] = {}
    best_auc = -1.0
    best_pr = -1.0
    for i, params in enumerate(candidates):
        pipe = _base_lr_pipeline(numeric_cols, categorical_cols)
        pipe.set_params(**params)
        pipe.fit(X_train, y_train)
        auc, pr = _score_val(pipe, X_val, y_val)
        history.append({"params": params, "val_roc_auc": auc, "val_pr_auc": pr})
        logger.info("LR tune [%d/%d] val_auc=%.4f params=%s", i + 1, len(candidates), auc, params)
        if auc > best_auc:
            best_auc, best_pr, best_pipe, best_params = auc, pr, pipe, params

    assert best_pipe is not None
    return TuneResult(
        best_pipeline=best_pipe,
        best_params=best_params,
        best_val_roc_auc=best_auc,
        best_val_pr_auc=best_pr,
        search_mode="holdout",
        n_candidates=len(candidates),
        history=history,
        elapsed_seconds=time.perf_counter() - t0,
    )


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_pos_weight: float,
    n_iter: int = 12,
    mode: str = "holdout",
    random_state: int = RANDOM_STATE,
) -> TuneResult:
    """Tune XGBoost hyperparameters; TEST must not be passed in."""
    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state + 7)
    history: list[dict[str, Any]] = []

    if mode == "cv":
        pipe = _base_xgb_pipeline(numeric_cols, categorical_cols, scale_pos_weight)
        param_dist = {
            "classifier__n_estimators": [150, 250, 350, 450],
            "classifier__max_depth": [3, 4, 5, 6, 7],
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "classifier__min_child_weight": [1, 3, 5, 10],
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "classifier__scale_pos_weight": [
                scale_pos_weight * 0.5,
                scale_pos_weight,
                scale_pos_weight * 1.5,
            ],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        val_auc, val_pr = _score_val(best, X_val, y_val)
        for i, params in enumerate(search.cv_results_["params"]):
            history.append(
                {
                    "params": {
                        k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                        for k, v in params.items()
                    },
                    "mean_cv_roc_auc": float(search.cv_results_["mean_test_score"][i]),
                }
            )
        return TuneResult(
            best_pipeline=best,
            best_params={
                k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                for k, v in search.best_params_.items()
            },
            best_val_roc_auc=val_auc,
            best_val_pr_auc=val_pr,
            search_mode="cv",
            n_candidates=n_iter,
            history=history,
            elapsed_seconds=time.perf_counter() - t0,
        )

    candidates = _xgb_param_space(rng, n_iter, scale_pos_weight)
    best_pipe: Optional[Pipeline] = None
    best_params: dict[str, Any] = {}
    best_auc = -1.0
    best_pr = -1.0
    for i, params in enumerate(candidates):
        pipe = _base_xgb_pipeline(numeric_cols, categorical_cols, scale_pos_weight)
        pipe.set_params(**params)
        pipe.fit(X_train, y_train)
        auc, pr = _score_val(pipe, X_val, y_val)
        history.append({"params": params, "val_roc_auc": auc, "val_pr_auc": pr})
        logger.info("XGB tune [%d/%d] val_auc=%.4f", i + 1, len(candidates), auc)
        if auc > best_auc:
            best_auc, best_pr, best_pipe, best_params = auc, pr, pipe, params

    assert best_pipe is not None
    return TuneResult(
        best_pipeline=best_pipe,
        best_params=best_params,
        best_val_roc_auc=best_auc,
        best_val_pr_auc=best_pr,
        search_mode="holdout",
        n_candidates=len(candidates),
        history=history,
        elapsed_seconds=time.perf_counter() - t0,
    )
