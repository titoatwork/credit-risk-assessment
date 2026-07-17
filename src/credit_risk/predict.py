"""Load artifacts and produce credit decisions + default probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from credit_risk.config import ARTIFACTS_PATH, DEFAULT_THRESHOLD, PREFERRED_MODEL
from credit_risk.features_multitable import add_application_derived_features
from credit_risk.preprocess import prepare_input_frame


class ArtifactBundle:
    """In-memory model bundle loaded from joblib."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.production_model: str = data.get("production_model", PREFERRED_MODEL)
        self.threshold: float = float(data.get("threshold", DEFAULT_THRESHOLD))
        self.feature_columns: list[str] = list(data["feature_columns"])
        self.column_descriptions: dict[str, str] = dict(data.get("column_descriptions") or {})
        self.transformed_feature_names: list[str] = list(
            data.get("transformed_feature_names") or []
        )
        self.background_X: Optional[pd.DataFrame] = data.get("background_X")
        self.metrics: dict[str, Any] = dict(data.get("metrics") or {})
        self.use_derived_features: bool = bool(data.get("use_derived_features", True))

    def get_pipeline(self, model_name: Optional[str] = None):
        name = model_name or self.production_model
        if name not in self.data:
            raise KeyError(f"Model {name!r} not in artifact bundle")
        return self.data[name]


def _features_to_frame(features: dict[str, Any], bundle: ArtifactBundle) -> pd.DataFrame:
    """Build a model-aligned single-row frame, recomputing derived ratios when needed."""
    # Start from raw keys; impute missing model columns later
    raw = dict(features)
    df = pd.DataFrame([raw])
    if bundle.use_derived_features:
        df = add_application_derived_features(df)
    return prepare_input_frame(df.iloc[0].to_dict(), bundle.feature_columns)


_BUNDLE: Optional[ArtifactBundle] = None


def load_bundle(path: Optional[Path] = None, force_reload: bool = False) -> ArtifactBundle:
    """Load (and cache) the trained artifact bundle."""
    global _BUNDLE
    if _BUNDLE is not None and not force_reload:
        return _BUNDLE
    art_path = Path(path) if path is not None else ARTIFACTS_PATH
    if not art_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found at {art_path}. Run training first: "
            "python -m credit_risk.train"
        )
    raw = joblib.load(art_path)
    _BUNDLE = ArtifactBundle(raw)
    return _BUNDLE


def reset_bundle_cache() -> None:
    """Clear cached bundle (for tests)."""
    global _BUNDLE
    _BUNDLE = None


def probability_to_decision(probability: float, threshold: float) -> str:
    """Map P(default) to approve/decline. High risk => decline."""
    if probability >= threshold:
        return "decline"
    return "approve"


def assess_applicant(
    features: dict[str, Any],
    bundle: Optional[ArtifactBundle] = None,
    model_name: Optional[str] = None,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    """
    Score one applicant.

    Returns dict with:
      decision: "approve" | "decline"
      default_probability: float in [0, 1]
      risk_score: same as default_probability (alias)
      model: model name used
      threshold: decision threshold
    """
    bundle = bundle or load_bundle()
    pipe = bundle.get_pipeline(model_name)
    thr = float(threshold if threshold is not None else bundle.threshold)

    X = _features_to_frame(features, bundle)
    proba = float(pipe.predict_proba(X)[0, 1])
    # Clamp numerical noise
    proba = float(np.clip(proba, 0.0, 1.0))
    decision = probability_to_decision(proba, thr)
    used_model = model_name or bundle.production_model

    return {
        "decision": decision,
        "default_probability": proba,
        "risk_score": proba,
        "model": used_model,
        "threshold": thr,
        "message": (
            "Application declined due to elevated default risk."
            if decision == "decline"
            else "Application approved based on assessed risk."
        ),
    }


def assess_dataframe(
    X: pd.DataFrame,
    bundle: Optional[ArtifactBundle] = None,
    model_name: Optional[str] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Batch score a DataFrame of applicants (columns = raw feature names)."""
    bundle = bundle or load_bundle()
    pipe = bundle.get_pipeline(model_name)
    thr = float(threshold if threshold is not None else bundle.threshold)

    # Recompute derived features for batch rows when the model expects them
    if bundle.use_derived_features:
        X = add_application_derived_features(X)
    aligned = X.reindex(columns=bundle.feature_columns)
    proba = pipe.predict_proba(aligned)[:, 1]
    decisions = np.where(proba >= thr, "decline", "approve")
    out = pd.DataFrame(
        {
            "decision": decisions,
            "default_probability": proba,
            "risk_score": proba,
        }
    )
    return out
