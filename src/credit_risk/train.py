"""
Train logistic regression and XGBoost with an academic ML lifecycle:

  EDA (optional) → feature engineering → stratified train/val/test →
  preprocess fit on train → baseline + candidates → select model on val →
  select threshold on val → final evaluation on test → save artifacts.

Threshold and production model are chosen on the VALIDATION set only;
reported operating metrics are from the untouched TEST set.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_risk.config import (
    ARTIFACTS_PATH,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_THRESHOLD,
    METRICS_PATH,
    PREFERRED_MODEL,
    RANDOM_STATE,
    TEST_SIZE,
)
from credit_risk.data import (
    load_application_train,
    load_column_descriptions,
    split_features_target,
)
from credit_risk.evaluation import full_model_report, ranking_metrics
from credit_risk.features_multitable import add_application_derived_features
from credit_risk.preprocess import build_preprocessor, get_feature_names, infer_feature_types

logger = logging.getLogger(__name__)

# Validation fraction of the temporary "train+val" pool (after holding out test)
VAL_SIZE = 0.2


@dataclass
class ModelMetrics:
    name: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    n_train: int
    n_test: int
    default_rate_train: float
    default_rate_test: float
    split: str = "test"


@dataclass
class TrainResult:
    metrics: list[ModelMetrics] = field(default_factory=list)
    production_model: str = "xgboost"
    threshold: float = DEFAULT_THRESHOLD
    feature_columns: list[str] = field(default_factory=list)
    artifacts_path: str = ""
    metrics_path: str = ""


def _evaluate(
    name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    n_train: int,
    n_test: int,
    default_rate_train: float,
    default_rate_test: float,
    threshold: float = DEFAULT_THRESHOLD,
    split: str = "test",
) -> ModelMetrics:
    pred = (proba >= threshold).astype(int)
    return ModelMetrics(
        name=name,
        roc_auc=float(roc_auc_score(y_true, proba)),
        pr_auc=float(average_precision_score(y_true, proba)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        n_train=n_train,
        n_test=n_test,
        default_rate_train=default_rate_train,
        default_rate_test=default_rate_test,
        split=split,
    )


def select_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    strategy: str = "youden",
) -> dict[str, float]:
    """Choose decision threshold on a single labeled split (use VALIDATION only)."""
    if strategy == "fixed":
        thr = 0.5
        pred = (proba >= thr).astype(int)
        return {
            "threshold": thr,
            "strategy": strategy,
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
        }

    if strategy == "f1":
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 37):
            pred = (proba >= thr).astype(int)
            f1 = float(f1_score(y_true, pred, zero_division=0))
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        pred = (proba >= best_thr).astype(int)
        return {
            "threshold": best_thr,
            "strategy": strategy,
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
        }

    fpr, tpr, thresholds = roc_curve(y_true, proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    best_thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    if not np.isfinite(best_thr):
        best_thr = 0.5
    best_thr = float(np.clip(best_thr, 0.01, 0.99))
    pred = (proba >= best_thr).astype(int)
    return {
        "threshold": best_thr,
        "strategy": "youden",
        "youden_j": float(j[idx]),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def build_logistic_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def build_xgboost_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_pos_weight: float,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def build_dummy_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Stratified random baseline — should ~ROC-AUC 0.5."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def train_models(
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE,
    data_path: Optional[Path] = None,
    artifacts_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    threshold: Optional[float] = None,
    threshold_strategy: str = "youden",
    random_state: int = RANDOM_STATE,
    preferred_model: str = PREFERRED_MODEL,
    multi_table: bool = False,
    run_eda: bool = True,
    tune: bool = False,
    tune_iter: int = 12,
    tune_mode: str = "holdout",
    refit_train_val: bool = True,
) -> TrainResult:
    """
    Academic train/eval lifecycle for LR + XGBoost (+ dummy baseline).

    Split protocol
    --------------
    1. Hold out stratified TEST (TEST_SIZE, default 20%).
    2. Split remaining into TRAIN / VAL (VAL_SIZE of remaining, default 20% → ~16% overall).
    3. Fit models on TRAIN only.
    4. Choose production model + threshold using VALIDATION scores only.
    5. Report final metrics on TEST only.
    """
    artifacts_path = Path(artifacts_path) if artifacts_path else ARTIFACTS_PATH
    metrics_path = Path(metrics_path) if metrics_path else METRICS_PATH
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)

    lifecycle_log: list[dict[str, Any]] = []

    # --- Stage 1–2: data ---
    logger.info("Stage: load data (sample_size=%s, multi_table=%s)", sample_size, multi_table)
    df = load_application_train(path=data_path, sample_size=sample_size, random_state=random_state)
    lifecycle_log.append({"stage": "data_load", "n_rows": int(len(df)), "sample_size": sample_size})

    # --- Stage 3: EDA ---
    if run_eda:
        try:
            from credit_risk.eda import run_eda as _run_eda

            eda_report = _run_eda(sample_size=sample_size, data_path=data_path)
            lifecycle_log.append(
                {
                    "stage": "eda",
                    "default_rate": eda_report.get("default_rate"),
                    "paths": eda_report.get("paths"),
                }
            )
            logger.info("EDA written to %s", eda_report.get("paths"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EDA stage skipped: %s", exc)
            lifecycle_log.append({"stage": "eda", "error": str(exc)})

    # --- Stage 4: feature engineering ---
    if multi_table:
        from credit_risk.features_multitable import build_multitable_features

        logger.info("Stage: multi-table feature engineering...")
        df = build_multitable_features(df)
    df = add_application_derived_features(df)
    X, y = split_features_target(df)
    feature_columns = list(X.columns)
    numeric_cols, categorical_cols = infer_feature_types(X)
    lifecycle_log.append(
        {
            "stage": "feature_engineering",
            "n_features": len(feature_columns),
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
            "multi_table": multi_table,
        }
    )
    logger.info(
        "Features=%d (num=%d cat=%d) default_rate=%.4f",
        len(feature_columns),
        len(numeric_cols),
        len(categorical_cols),
        float(y.mean()),
    )

    # --- Stage 5: stratified train / val / test ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=random_state,
        stratify=y_trainval,
    )
    lifecycle_log.append(
        {
            "stage": "split",
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "default_rate_train": float(y_train.mean()),
            "default_rate_val": float(y_val.mean()),
            "default_rate_test": float(y_test.mean()),
            "protocol": "stratified train/val/test; model+threshold on val; final metrics on test",
        }
    )
    logger.info(
        "Split train/val/test = %d / %d / %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    # --- Stage 6–8: baseline + candidates (optional hyperparameter tuning) ---
    logger.info("Stage: train Dummy baseline...")
    dummy_pipe = build_dummy_pipeline(numeric_cols, categorical_cols)
    dummy_pipe.fit(X_train, y_train)

    tuning_report: dict[str, Any] = {"enabled": bool(tune), "mode": tune_mode if tune else None}

    if tune:
        from credit_risk.tuning import tune_logistic_regression, tune_xgboost

        logger.info(
            "Stage: hyperparameter tuning (mode=%s, n_iter≈%d) on TRAIN; score on VAL — TEST unused",
            tune_mode,
            tune_iter,
        )
        lr_tune = tune_logistic_regression(
            X_train,
            y_train,
            X_val,
            y_val,
            numeric_cols,
            categorical_cols,
            n_iter=max(4, tune_iter // 2),
            mode=tune_mode,
            random_state=random_state,
        )
        xgb_tune = tune_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            numeric_cols,
            categorical_cols,
            scale_pos_weight,
            n_iter=tune_iter,
            mode=tune_mode,
            random_state=random_state,
        )
        lr_pipe = lr_tune.best_pipeline
        xgb_pipe = xgb_tune.best_pipeline
        tuning_report.update(
            {
                "logistic_regression": {
                    "best_params": lr_tune.best_params,
                    "best_val_roc_auc": lr_tune.best_val_roc_auc,
                    "best_val_pr_auc": lr_tune.best_val_pr_auc,
                    "n_candidates": lr_tune.n_candidates,
                    "search_mode": lr_tune.search_mode,
                    "elapsed_seconds": lr_tune.elapsed_seconds,
                    "history": lr_tune.history,
                },
                "xgboost": {
                    "best_params": xgb_tune.best_params,
                    "best_val_roc_auc": xgb_tune.best_val_roc_auc,
                    "best_val_pr_auc": xgb_tune.best_val_pr_auc,
                    "n_candidates": xgb_tune.n_candidates,
                    "search_mode": xgb_tune.search_mode,
                    "elapsed_seconds": xgb_tune.elapsed_seconds,
                    "history": xgb_tune.history,
                },
            }
        )
        logger.info(
            "Tune done | LR best val AUC=%.4f | XGB best val AUC=%.4f",
            lr_tune.best_val_roc_auc,
            xgb_tune.best_val_roc_auc,
        )
    else:
        logger.info("Stage: train logistic regression (default hyperparameters)...")
        lr_pipe = build_logistic_pipeline(numeric_cols, categorical_cols)
        lr_pipe.fit(X_train, y_train)

        logger.info("Stage: train XGBoost default (scale_pos_weight=%.3f)...", scale_pos_weight)
        xgb_pipe = build_xgboost_pipeline(numeric_cols, categorical_cols, scale_pos_weight)
        xgb_pipe.fit(X_train, y_train)

    # Validation scores (model selection + threshold) — after tuning pipelines are already fit on train
    dummy_val = dummy_pipe.predict_proba(X_val)[:, 1]
    lr_val = lr_pipe.predict_proba(X_val)[:, 1]
    xgb_val = xgb_pipe.predict_proba(X_val)[:, 1]

    lr_val_auc = float(roc_auc_score(y_val.values, lr_val))
    xgb_val_auc = float(roc_auc_score(y_val.values, xgb_val))
    dummy_val_auc = float(roc_auc_score(y_val.values, dummy_val))
    lr_val_pr = float(average_precision_score(y_val.values, lr_val))
    xgb_val_pr = float(average_precision_score(y_val.values, xgb_val))
    logger.info(
        "VAL ROC-AUC  dummy=%.4f  LR=%.4f  XGB=%.4f",
        dummy_val_auc,
        lr_val_auc,
        xgb_val_auc,
    )

    # --- Stage 9: production model on VALIDATION ---
    if preferred_model == "logistic_regression":
        production = "logistic_regression"
    elif preferred_model == "xgboost":
        production = "xgboost"
    else:
        if abs(xgb_val_auc - lr_val_auc) < 0.005:
            production = "xgboost" if xgb_val_pr >= lr_val_pr else "logistic_regression"
        else:
            production = "xgboost" if xgb_val_auc >= lr_val_auc else "logistic_regression"

    prod_val = xgb_val if production == "xgboost" else lr_val

    # --- Stage 10: threshold on VALIDATION only ---
    if threshold is None:
        if threshold_strategy == "fixed":
            thr_info = select_threshold(y_val.values, prod_val, strategy="fixed")
            threshold = DEFAULT_THRESHOLD
            thr_info["threshold"] = threshold
        else:
            thr_info = select_threshold(y_val.values, prod_val, strategy=threshold_strategy)
            threshold = float(thr_info["threshold"])
        thr_info["selected_on"] = "validation"
        logger.info(
            "Threshold=%.4f via %s on VALIDATION (f1=%.4f)",
            threshold,
            thr_info.get("strategy"),
            thr_info.get("f1", 0),
        )
    else:
        thr_info = {
            "threshold": float(threshold),
            "strategy": "manual",
            "selected_on": "manual",
        }

    lifecycle_log.append(
        {
            "stage": "model_selection_validation",
            "production_model": production,
            "val_roc_auc": {
                "dummy": dummy_val_auc,
                "logistic_regression": lr_val_auc,
                "xgboost": xgb_val_auc,
            },
            "val_pr_auc": {
                "logistic_regression": lr_val_pr,
                "xgboost": xgb_val_pr,
            },
            "threshold_info": thr_info,
            "hyperparameter_tuning": tuning_report,
        }
    )

    # Optional academic step: after hyperparameters & model choice, refit production
    # candidates on train+val so final test eval uses more data (params already fixed on val).
    if refit_train_val:
        logger.info("Stage: refit LR + XGB on train+val with selected hyperparameters...")
        X_tv = pd.concat([X_train, X_val], axis=0)
        y_tv = pd.concat([y_train, y_val], axis=0)
        # Clone params from tuned/default pipelines
        lr_params = {
            k: v
            for k, v in lr_pipe.get_params().items()
            if k.startswith("classifier__")
        }
        xgb_params = {
            k: v
            for k, v in xgb_pipe.get_params().items()
            if k.startswith("classifier__")
        }
        lr_pipe = build_logistic_pipeline(numeric_cols, categorical_cols)
        lr_pipe.set_params(**{k: v for k, v in lr_params.items() if k in lr_pipe.get_params()})
        lr_pipe.fit(X_tv, y_tv)
        xgb_pipe = build_xgboost_pipeline(numeric_cols, categorical_cols, scale_pos_weight)
        # apply tuned xgb params (including scale_pos_weight if tuned)
        safe_xgb = {k: v for k, v in xgb_params.items() if k in xgb_pipe.get_params()}
        xgb_pipe.set_params(**safe_xgb)
        xgb_pipe.fit(X_tv, y_tv)
        # Dummy stays train-only (baseline fairness); threshold already fixed from val
        lifecycle_log.append(
            {
                "stage": "refit_train_val",
                "n_train_val": int(len(X_tv)),
                "note": "Hyperparameters and threshold fixed before refit; test still untouched",
            }
        )

    # --- Stage 11: final evaluation on TEST only ---
    dummy_test = dummy_pipe.predict_proba(X_test)[:, 1]
    lr_test = lr_pipe.predict_proba(X_test)[:, 1]
    xgb_test = xgb_pipe.predict_proba(X_test)[:, 1]

    dummy_metrics = _evaluate(
        "dummy_stratified",
        y_test.values,
        dummy_test,
        n_train=len(y_train),
        n_test=len(y_test),
        default_rate_train=float(y_train.mean()),
        default_rate_test=float(y_test.mean()),
        threshold=threshold,
        split="test",
    )
    lr_metrics = _evaluate(
        "logistic_regression",
        y_test.values,
        lr_test,
        n_train=len(y_train),
        n_test=len(y_test),
        default_rate_train=float(y_train.mean()),
        default_rate_test=float(y_test.mean()),
        threshold=threshold,
        split="test",
    )
    xgb_metrics = _evaluate(
        "xgboost",
        y_test.values,
        xgb_test,
        n_train=len(y_train),
        n_test=len(y_test),
        default_rate_train=float(y_train.mean()),
        default_rate_test=float(y_test.mean()),
        threshold=threshold,
        split="test",
    )
    logger.info(
        "TEST ROC-AUC  dummy=%.4f  LR=%.4f  XGB=%.4f  (production=%s)",
        dummy_metrics.roc_auc,
        lr_metrics.roc_auc,
        xgb_metrics.roc_auc,
        production,
    )

    lr_report = full_model_report("logistic_regression", y_test.values, lr_test, threshold)
    xgb_report = full_model_report("xgboost", y_test.values, xgb_test, threshold)
    dummy_report = full_model_report("dummy_stratified", y_test.values, dummy_test, threshold)
    lr_rank = ranking_metrics(y_test.values, lr_test)
    xgb_rank = ranking_metrics(y_test.values, xgb_test)
    dummy_rank = ranking_metrics(y_test.values, dummy_test)

    # --- Persist ---
    prod_pipe = xgb_pipe if production == "xgboost" else lr_pipe
    transformed_feature_names = get_feature_names(prod_pipe.named_steps["preprocessor"])
    column_descriptions = load_column_descriptions()
    bg_n = min(200, len(X_train))
    background_X = X_train.sample(n=bg_n, random_state=random_state)

    bundle: dict[str, Any] = {
        "logistic_regression": lr_pipe,
        "xgboost": xgb_pipe,
        "dummy_stratified": dummy_pipe,
        "production_model": production,
        "threshold": threshold,
        "threshold_info": thr_info,
        "feature_columns": feature_columns,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "transformed_feature_names": transformed_feature_names,
        "column_descriptions": column_descriptions,
        "background_X": background_X,
        "use_derived_features": True,
        "multi_table": multi_table,
        "metrics": {
            "dummy_stratified": asdict(dummy_metrics),
            "logistic_regression": asdict(lr_metrics),
            "xgboost": asdict(xgb_metrics),
        },
        "split_protocol": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
            "threshold_selected_on": "validation",
            "final_metrics_on": "test",
        },
        "lifecycle": "academic_v2_train_val_test",
        "random_state": random_state,
        "sample_size": sample_size,
    }
    joblib.dump(bundle, artifacts_path)
    logger.info("Saved artifacts -> %s", artifacts_path)

    metrics_payload = {
        "lifecycle": "academic_v2_train_val_test",
        "stages": lifecycle_log,
        "hyperparameter_tuning": tuning_report,
        "production_model": production,
        "threshold": threshold,
        "threshold_info": thr_info,
        "sample_size": sample_size,
        "n_rows_used": int(len(X)),
        "n_features": int(len(feature_columns)),
        "multi_table": multi_table,
        "split": {
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_val)),
            "n_test": int(len(X_test)),
            "threshold_selected_on": "validation",
            "model_selected_on": "validation",
            "final_metrics_on": "test",
        },
        "base_rate_test": float(y_test.mean()),
        "validation_ranking": {
            "dummy_roc_auc": dummy_val_auc,
            "logistic_regression_roc_auc": lr_val_auc,
            "xgboost_roc_auc": xgb_val_auc,
            "logistic_regression_pr_auc": lr_val_pr,
            "xgboost_pr_auc": xgb_val_pr,
        },
        "models": {
            "dummy_stratified": {
                **asdict(dummy_metrics),
                "ranking": dummy_rank,
                "operating_point": dummy_report["operating_point"],
            },
            "logistic_regression": {
                **asdict(lr_metrics),
                "ranking": lr_rank,
                "operating_point": lr_report["operating_point"],
            },
            "xgboost": {
                **asdict(xgb_metrics),
                "ranking": xgb_rank,
                "operating_point": xgb_report["operating_point"],
            },
        },
        "threshold_sweep": {
            "logistic_regression": lr_report["threshold_sweep"],
            "xgboost": xgb_report["threshold_sweep"],
        },
        "beats_baseline": {
            "logistic_regression_roc_auc_gt_dummy": lr_metrics.roc_auc > dummy_metrics.roc_auc + 0.05,
            "xgboost_roc_auc_gt_dummy": xgb_metrics.roc_auc > dummy_metrics.roc_auc + 0.05,
        },
        "skeptics_summary": {
            "why_precision_can_look_low": (
                "With base_rate≈8%, precision at a high-recall threshold is modest; "
                "compare lift_vs_base_rate and ROC-AUC vs dummy baseline."
            ),
            "primary_skill_metrics": ["roc_auc", "pr_auc", "lift_vs_base_rate", "precision_at_top_10pct"],
            "academic_protocol": "train/val/test; threshold and model selection on validation only",
        },
        "note": (
            "Academic lifecycle: stratified train/val/test. "
            "Models fit on train; production model and threshold chosen on validation; "
            "reported ROC-AUC/PR-AUC/precision/recall/lift are on untouched test. "
            "Dummy stratified baseline included. "
            "TARGET=1 = payment difficulties; decline if P(default) >= threshold."
        ),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Saved metrics -> %s", metrics_path)

    summary_path = metrics_path.parent / "metrics_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Model metrics summary (academic train/val/test)",
                "",
                f"- Production model: **{production}**",
                f"- Rows used: **{len(X)}** (sample_size={sample_size})",
                f"- Split: train **{len(X_train)}** / val **{len(X_val)}** / test **{len(X_test)}**",
                f"- Threshold selected on: **validation** (tau={threshold:.4f}, {thr_info.get('strategy')})",
                f"- Final metrics on: **test**",
                f"- Multi-table: **{multi_table}** | Features: **{len(feature_columns)}**",
                f"- Test base rate: **{float(y_test.mean()):.2%}**",
                "",
                "| Model | Split | ROC-AUC | PR-AUC | Precision@tau | Recall@tau | Lift@tau | Prec@top10% |",
                "|-------|-------|---------|--------|---------------|------------|----------|-------------|",
                (
                    f"| Dummy (stratified) | test | {dummy_metrics.roc_auc:.4f} | {dummy_metrics.pr_auc:.4f} | "
                    f"{dummy_metrics.precision:.4f} | {dummy_metrics.recall:.4f} | "
                    f"{dummy_report['operating_point']['lift_vs_base_rate']:.2f}x | "
                    f"{dummy_rank.get('precision_at_top_10pct', 0):.4f} |"
                ),
                (
                    f"| Logistic Regression | test | {lr_metrics.roc_auc:.4f} | {lr_metrics.pr_auc:.4f} | "
                    f"{lr_metrics.precision:.4f} | {lr_metrics.recall:.4f} | "
                    f"{lr_report['operating_point']['lift_vs_base_rate']:.2f}x | "
                    f"{lr_rank.get('precision_at_top_10pct', 0):.4f} |"
                ),
                (
                    f"| XGBoost | test | {xgb_metrics.roc_auc:.4f} | {xgb_metrics.pr_auc:.4f} | "
                    f"{xgb_metrics.precision:.4f} | {xgb_metrics.recall:.4f} | "
                    f"{xgb_report['operating_point']['lift_vs_base_rate']:.2f}x | "
                    f"{xgb_rank.get('precision_at_top_10pct', 0):.4f} |"
                ),
                "",
                "## Academic checklist",
                "",
                "- [x] Problem definition (default risk -> approve/decline)",
                "- [x] EDA report (if enabled)",
                "- [x] Feature engineering + Pipeline preprocessing",
                "- [x] Stratified train/validation/test",
                "- [x] Baseline dummy model",
                "- [x] LR + XGBoost with imbalance handling",
                "- [x] Model selection on validation",
                "- [x] Threshold selection on validation",
                "- [x] Final metrics on held-out test",
                "- [x] SHAP explainability + API/dashboard (separate modules)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Saved summary -> %s", summary_path)

    return TrainResult(
        metrics=[dummy_metrics, lr_metrics, xgb_metrics],
        production_model=production,
        threshold=threshold,
        feature_columns=feature_columns,
        artifacts_path=str(artifacts_path),
        metrics_path=str(metrics_path),
    )


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Academic ML lifecycle: train LR + XGBoost (train/val/test protocol)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Rows to sample (0 = full dataset).",
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--artifacts", type=str, default=str(ARTIFACTS_PATH))
    parser.add_argument("--metrics", type=str, default=str(METRICS_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        default="youden",
        choices=["youden", "f1", "fixed"],
    )
    parser.add_argument("--multi-table", action="store_true")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA report generation")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter search (fit on train, score on validation; never on test)",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=12,
        help="Approx. number of XGBoost random configs (LR uses ~half)",
    )
    parser.add_argument(
        "--tune-mode",
        type=str,
        default="holdout",
        choices=["holdout", "cv"],
        help="holdout = score each config on validation; cv = RandomizedSearchCV on train",
    )
    parser.add_argument(
        "--no-refit-train-val",
        action="store_true",
        help="Do not refit selected models on train+val before test evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PREFERRED_MODEL,
        choices=["xgboost", "logistic_regression", "auto"],
    )
    args = parser.parse_args(argv)

    sample = None if args.sample_size == 0 else args.sample_size
    thr = args.threshold
    if thr is None and args.threshold_strategy == "fixed":
        thr = DEFAULT_THRESHOLD
    result = train_models(
        sample_size=sample,
        data_path=Path(args.data_path) if args.data_path else None,
        artifacts_path=Path(args.artifacts),
        metrics_path=Path(args.metrics),
        threshold=thr,
        threshold_strategy=args.threshold_strategy,
        multi_table=args.multi_table,
        preferred_model=args.model,
        run_eda=not args.no_eda,
        tune=args.tune,
        tune_iter=args.tune_iter,
        tune_mode=args.tune_mode,
        refit_train_val=not args.no_refit_train_val,
    )
    print(
        json.dumps(
            {
                "production_model": result.production_model,
                "threshold": result.threshold,
                "metrics": [asdict(m) for m in result.metrics],
                "artifacts_path": result.artifacts_path,
                "metrics_path": result.metrics_path,
                "lifecycle": "academic_v2_train_val_test",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
