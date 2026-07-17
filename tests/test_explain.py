"""SHAP explain path on shipped explain_applicant for a declined sample."""

from __future__ import annotations

from credit_risk.explain import explain_applicant, explain_decline
from credit_risk.predict import assess_applicant


def test_shap_top_features_for_high_risk(bundle, high_risk_features):
    assessment = assess_applicant(high_risk_features, bundle=bundle, threshold=0.0)
    assert assessment["decision"] == "decline"

    explanation = explain_applicant(
        high_risk_features,
        bundle=bundle,
        model_name=assessment["model"],
        top_k=5,
    )
    assert explanation["top_features"], "SHAP top_features must be non-empty"
    assert len(explanation["top_features"]) <= 5

    model_feats = set(bundle.feature_columns) | set(bundle.transformed_feature_names)
    for item in explanation["top_features"]:
        assert "feature" in item
        assert "shap_value" in item or "contribution" in item
        assert item["direction"] in {"increased_risk", "decreased_risk", "neutral"}
        # Feature should relate to model feature set
        assert (
            item["feature"] in model_feats
            or item.get("transformed_feature") in model_feats
            or any(
                str(item.get("transformed_feature", "")).startswith(c + "_")
                or item["feature"] == c
                for c in bundle.feature_columns
            )
        )

    full = explain_decline(high_risk_features, assessment, bundle=bundle, top_k=5)
    assert full["decision"] == "decline"
    assert full["decline_reasons"]
    assert full["explanation"]["top_features"]
