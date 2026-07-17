"""SHAP-based local explanations for credit decline decisions."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from credit_risk.config import SHAP_TOP_K
from credit_risk.predict import ArtifactBundle, load_bundle
from credit_risk.preprocess import prepare_input_frame


def _original_feature_name(transformed_name: str, raw_columns: list[str]) -> str:
    """
    Map one-hot / transformed name back toward a raw column when possible.
    OneHotEncoder names look like CATEGORY_value; numeric often stay as-is.
    """
    if transformed_name in raw_columns:
        return transformed_name
    # Try longest prefix match against raw columns
    best = transformed_name
    best_len = 0
    for col in raw_columns:
        if transformed_name.startswith(col + "_") and len(col) > best_len:
            best = col
            best_len = len(col)
        elif transformed_name == col:
            return col
    return best


def _direction_label(shap_value: float) -> str:
    """Positive SHAP => increased predicted default risk (for risk models)."""
    if shap_value > 0:
        return "increased_risk"
    if shap_value < 0:
        return "decreased_risk"
    return "neutral"


def _reason_text(
    feature: str,
    shap_value: float,
    description: Optional[str],
) -> str:
    direction = "raised" if shap_value > 0 else "lowered"
    base = description or feature
    return (
        f"{base} ({feature}) {direction} the predicted default risk "
        f"(SHAP contribution: {shap_value:+.4f})."
    )


def explain_applicant(
    features: dict[str, Any],
    bundle: Optional[ArtifactBundle] = None,
    model_name: Optional[str] = None,
    top_k: int = SHAP_TOP_K,
) -> dict[str, Any]:
    """
    Compute local SHAP explanation for one applicant.

    Uses TreeExplainer for XGBoost; LinearExplainer (or coefficient fallback)
    for logistic regression. Returns top features by |SHAP| with direction.
    """
    import shap

    bundle = bundle or load_bundle()
    used_model = model_name or bundle.production_model
    pipe = bundle.get_pipeline(used_model)
    preprocessor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]

    X = prepare_input_frame(features, bundle.feature_columns)
    X_t = preprocessor.transform(X)

    feature_names = list(bundle.transformed_feature_names)
    if not feature_names:
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(X_t.shape[1])]

    # Ensure 2d dense
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_t = np.asarray(X_t, dtype=float)

    shap_values_row: np.ndarray
    base_value: float

    if used_model == "xgboost" or type(classifier).__name__.startswith("XGB"):
        explainer = shap.TreeExplainer(classifier)
        sv = explainer.shap_values(X_t)
        # Binary: shap_values may be (n, n_features) or list
        if isinstance(sv, list):
            shap_values_row = np.asarray(sv[1][0], dtype=float)
        else:
            shap_values_row = np.asarray(sv[0], dtype=float)
        bv = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            base_value = float(np.asarray(bv).ravel()[-1])
        else:
            base_value = float(bv)
    else:
        # Logistic regression: try LinearExplainer with background
        bg = bundle.background_X
        if bg is not None and len(bg) > 0:
            bg_t = preprocessor.transform(bg.reindex(columns=bundle.feature_columns))
            if hasattr(bg_t, "toarray"):
                bg_t = bg_t.toarray()
            bg_t = np.asarray(bg_t, dtype=float)
            # Subsample background for speed
            if len(bg_t) > 100:
                idx = np.random.RandomState(42).choice(len(bg_t), 100, replace=False)
                bg_t = bg_t[idx]
            try:
                explainer = shap.LinearExplainer(classifier, bg_t)
                sv = explainer.shap_values(X_t)
                if isinstance(sv, list):
                    shap_values_row = np.asarray(sv[1][0], dtype=float)
                else:
                    shap_values_row = np.asarray(sv[0], dtype=float)
                bv = explainer.expected_value
                if isinstance(bv, (list, np.ndarray)):
                    base_value = float(np.asarray(bv).ravel()[-1])
                else:
                    base_value = float(bv)
            except Exception:
                # Coefficient * centered feature fallback
                coef = np.asarray(classifier.coef_).ravel()
                mean_bg = bg_t.mean(axis=0)
                shap_values_row = coef * (X_t[0] - mean_bg)
                base_value = float(classifier.intercept_.ravel()[0] + float((coef * mean_bg).sum()))
        else:
            coef = np.asarray(classifier.coef_).ravel()
            shap_values_row = coef * X_t[0]
            base_value = float(classifier.intercept_.ravel()[0])

    # Align lengths
    n = min(len(shap_values_row), len(feature_names))
    shap_values_row = shap_values_row[:n]
    feature_names = feature_names[:n]

    order = np.argsort(-np.abs(shap_values_row))[: max(1, top_k)]
    top_features: list[dict[str, Any]] = []
    for i in order:
        tname = feature_names[i]
        raw_name = _original_feature_name(tname, bundle.feature_columns)
        sv_i = float(shap_values_row[i])
        desc = bundle.column_descriptions.get(raw_name) or bundle.column_descriptions.get(tname)
        top_features.append(
            {
                "feature": raw_name,
                "transformed_feature": tname,
                "shap_value": sv_i,
                "contribution": sv_i,
                "direction": _direction_label(sv_i),
                "description": desc,
                "reason": _reason_text(raw_name, sv_i, desc),
            }
        )

    # Human-readable decline reasons: features that increased risk
    decline_reasons = [
        f["reason"]
        for f in top_features
        if f["direction"] == "increased_risk"
    ]
    if not decline_reasons and top_features:
        # Still surface top absolute contributors
        decline_reasons = [f["reason"] for f in top_features[:3]]

    return {
        "model": used_model,
        "base_value": base_value,
        "top_features": top_features,
        "decline_reasons": decline_reasons,
        "explanation_summary": (
            "Top factors influencing the default-risk prediction: "
            + "; ".join(f["feature"] + f" ({f['direction']})" for f in top_features[:5])
        ),
    }


def explain_decline(
    features: dict[str, Any],
    assessment: dict[str, Any],
    bundle: Optional[ArtifactBundle] = None,
    top_k: int = SHAP_TOP_K,
) -> dict[str, Any]:
    """
    Full assess + explain response for API. Always includes assessment;
    SHAP reasons are populated especially for declines.
    """
    bundle = bundle or load_bundle()
    model_name = assessment.get("model")
    explanation = explain_applicant(
        features,
        bundle=bundle,
        model_name=model_name,
        top_k=top_k,
    )
    return {
        **assessment,
        "explanation": explanation,
        "decline_reasons": (
            explanation["decline_reasons"]
            if assessment.get("decision") == "decline"
            else []
        ),
    }
