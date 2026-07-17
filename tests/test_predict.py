"""Predict path: decision + probability from shipped assess_applicant."""

from __future__ import annotations

import pytest

from credit_risk.predict import assess_applicant, probability_to_decision


def test_probability_to_decision_threshold():
    assert probability_to_decision(0.49, 0.5) == "approve"
    assert probability_to_decision(0.5, 0.5) == "decline"
    assert probability_to_decision(0.9, 0.5) == "decline"


def test_assess_returns_decision_and_probability(bundle, high_risk_features, low_risk_features):
    high = assess_applicant(high_risk_features, bundle=bundle)
    low = assess_applicant(low_risk_features, bundle=bundle)

    for out in (high, low):
        assert "decision" in out
        assert out["decision"] in {"approve", "decline"}
        assert 0.0 <= out["default_probability"] <= 1.0
        assert out["risk_score"] == out["default_probability"]
        assert out["threshold"] == bundle.threshold
        # Decision must match threshold rule on the same probability
        expected = probability_to_decision(out["default_probability"], out["threshold"])
        assert out["decision"] == expected

    # High-risk fixture should not score safer than low-risk fixture
    assert high["default_probability"] >= low["default_probability"] - 1e-9


def test_forced_decline_via_low_threshold(bundle, low_risk_features):
    """Even a safer applicant declines if threshold is extremely low."""
    out = assess_applicant(low_risk_features, bundle=bundle, threshold=0.0)
    assert out["decision"] == "decline"
    assert out["default_probability"] >= 0.0
