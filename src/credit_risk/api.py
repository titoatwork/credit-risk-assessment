"""FastAPI service for credit risk assessment + SHAP decline explanations."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from credit_risk.config import ARTIFACTS_PATH, METRICS_PATH
from credit_risk.explain import explain_applicant, explain_decline
from credit_risk.predict import assess_applicant, load_bundle, reset_bundle_cache
from credit_risk.schemas import (
    AssessRequest,
    AssessResponse,
    HealthResponse,
    MetricsResponse,
)

logger = logging.getLogger(__name__)


def create_app(artifacts_path: Optional[str] = None) -> FastAPI:
    """Application factory so tests can point at a specific artifact bundle."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = artifacts_path or str(ARTIFACTS_PATH)
        try:
            reset_bundle_cache()
            bundle = load_bundle(path=path, force_reload=True)
            app.state.bundle = bundle
            app.state.artifacts_path = path
            app.state.load_error = None
            logger.info(
                "Loaded model bundle from %s (production=%s)",
                path,
                bundle.production_model,
            )
        except Exception as exc:  # noqa: BLE001 — surface load failure via /health
            app.state.bundle = None
            app.state.artifacts_path = path
            app.state.load_error = str(exc)
            logger.warning("Failed to load artifacts at startup: %s", exc)
        yield
        reset_bundle_cache()

    app = FastAPI(
        title="Credit Risk Assessment API",
        description=(
            "Assess credit card applications using logistic regression and XGBoost. "
            "Declines include SHAP-based reasons for elevated default risk. "
            "Home Credit TARGET maps to default risk → approve/decline via threshold."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def _require_bundle():
        bundle = getattr(app.state, "bundle", None)
        if bundle is None:
            # Attempt lazy load
            try:
                path = getattr(app.state, "artifacts_path", None) or str(ARTIFACTS_PATH)
                bundle = load_bundle(path=path, force_reload=True)
                app.state.bundle = bundle
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=503,
                    detail=f"Model artifacts unavailable: {exc}",
                ) from exc
        return bundle

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        bundle = getattr(app.state, "bundle", None)
        if bundle is None:
            return HealthResponse(
                status="degraded" if getattr(app.state, "load_error", None) else "ok",
                model_loaded=False,
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            production_model=bundle.production_model,
            threshold=bundle.threshold,
            n_features=len(bundle.feature_columns),
            metrics=bundle.metrics,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> Any:
        bundle = _require_bundle()
        if METRICS_PATH.exists():
            import json

            payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            return payload
        return {
            "production_model": bundle.production_model,
            "threshold": bundle.threshold,
            "models": bundle.metrics,
            "note": "Metrics from in-memory artifact bundle",
        }

    @app.get("/features")
    def list_features() -> dict[str, Any]:
        """Return expected raw feature column names for request payloads."""
        bundle = _require_bundle()
        return {
            "feature_columns": bundle.feature_columns,
            "n_features": len(bundle.feature_columns),
            "descriptions": {
                k: v
                for k, v in bundle.column_descriptions.items()
                if k in set(bundle.feature_columns)
            },
        }

    @app.post("/assess", response_model=AssessResponse)
    def assess(req: AssessRequest) -> Any:
        """
        Score an applicant: decision + default probability.
        For declines (or when include_explanation=true), attach SHAP reasons.
        """
        bundle = _require_bundle()
        try:
            assessment = assess_applicant(
                req.features,
                bundle=bundle,
                model_name=req.model,
                threshold=req.threshold,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Assessment failed: {exc}") from exc

        need_explain = req.include_explanation or assessment["decision"] == "decline"
        if need_explain:
            try:
                full = explain_decline(
                    req.features,
                    assessment,
                    bundle=bundle,
                    top_k=req.top_k,
                )
                return full
            except Exception as exc:  # noqa: BLE001
                logger.exception("SHAP explanation failed")
                # Still return assessment without explanation
                assessment["decline_reasons"] = []
                assessment["explanation"] = None
                assessment["explanation_error"] = str(exc)
                return assessment

        assessment["decline_reasons"] = []
        assessment["explanation"] = None
        return assessment

    @app.post("/explain", response_model=AssessResponse)
    def explain(req: AssessRequest) -> Any:
        """Force assessment + full SHAP explanation (even for approvals)."""
        bundle = _require_bundle()
        try:
            assessment = assess_applicant(
                req.features,
                bundle=bundle,
                model_name=req.model,
                threshold=req.threshold,
            )
            full = explain_decline(
                req.features,
                assessment,
                bundle=bundle,
                top_k=req.top_k,
            )
            # Always include reasons on this endpoint
            full["decline_reasons"] = full.get("explanation", {}).get("decline_reasons", [])
            return full
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Explain failed: {exc}") from exc

    @app.exception_handler(Exception)
    async def unhandled(request, exc):  # type: ignore[no-untyped-def]
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


# Default app instance for uvicorn credit_risk.api:app
app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "credit_risk.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
