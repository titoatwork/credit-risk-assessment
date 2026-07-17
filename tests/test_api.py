"""In-process API tests against the real FastAPI app and trained bundle."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credit_risk.api import create_app
from credit_risk.predict import reset_bundle_cache


@pytest.fixture
def client(trained_bundle):
    reset_bundle_cache()
    app = create_app(artifacts_path=str(trained_bundle["artifacts_path"]))
    with TestClient(app) as c:
        yield c
    reset_bundle_cache()


def test_health_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["production_model"] in {"xgboost", "logistic_regression"}
    assert body["n_features"] > 0


def test_assess_valid_payload(client, high_risk_features):
    r = client.post(
        "/assess",
        json={
            "features": high_risk_features,
            "include_explanation": True,
            "threshold": 0.0,  # force decline path + explanations
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "decline"
    assert 0.0 <= body["default_probability"] <= 1.0
    assert "threshold" in body
    assert body.get("decline_reasons") is not None
    # With include_explanation / decline, explanation should be present
    if body.get("explanation"):
        assert body["explanation"]["top_features"]


def test_assess_rejects_empty_features(client):
    r = client.post("/assess", json={"features": {}})
    assert r.status_code == 422

    r2 = client.post("/assess", json={"features": {"AMT_INCOME_TOTAL": None, "FOO": None}})
    assert r2.status_code == 422


def test_assess_rejects_bad_model(client, low_risk_features):
    r = client.post(
        "/assess",
        json={"features": low_risk_features, "model": "not_a_model"},
    )
    assert r.status_code == 422


def test_explain_endpoint(client, high_risk_features):
    r = client.post(
        "/explain",
        json={"features": high_risk_features, "threshold": 0.0, "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "decision" in body
    assert "default_probability" in body
    assert body.get("explanation") is not None
    assert body["explanation"]["top_features"]
