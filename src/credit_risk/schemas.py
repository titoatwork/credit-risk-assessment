"""Pydantic request/response models for the credit risk API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ApplicantFeatures(BaseModel):
    """
    Applicant feature payload. Accepts any Home Credit application columns
    as extra fields. At least one feature must be provided.
    """

    model_config = {"extra": "allow"}

    # Common numeric examples (optional hints for OpenAPI; not required)
    AMT_INCOME_TOTAL: Optional[float] = Field(default=None, description="Income of the client")
    AMT_CREDIT: Optional[float] = Field(default=None, description="Credit amount of the loan")
    AMT_ANNUITY: Optional[float] = Field(default=None, description="Loan annuity")
    DAYS_BIRTH: Optional[int] = Field(
        default=None, description="Client age in days at application (negative)"
    )
    DAYS_EMPLOYED: Optional[int] = Field(default=None, description="Days employed (negative)")
    CODE_GENDER: Optional[str] = Field(default=None, description="Gender code M/F")
    NAME_EDUCATION_TYPE: Optional[str] = Field(default=None)
    NAME_FAMILY_STATUS: Optional[str] = Field(default=None)
    NAME_HOUSING_TYPE: Optional[str] = Field(default=None)
    NAME_INCOME_TYPE: Optional[str] = Field(default=None)
    FLAG_OWN_CAR: Optional[str] = Field(default=None)
    FLAG_OWN_REALTY: Optional[str] = Field(default=None)
    CNT_CHILDREN: Optional[int] = Field(default=None, ge=0)
    EXT_SOURCE_1: Optional[float] = Field(default=None)
    EXT_SOURCE_2: Optional[float] = Field(default=None)
    EXT_SOURCE_3: Optional[float] = Field(default=None)

    @field_validator("CODE_GENDER")
    @classmethod
    def _gender(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().upper()
        if v not in {"M", "F", "XNA"}:
            # Allow but warn via API validation only for clearly empty
            pass
        return v

    def to_feature_dict(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=False)
        # Drop keys that are explicitly None so missing -> NaN in prepare_input_frame
        # Keep all keys from model + extras; None becomes missing
        return {k: v for k, v in data.items()}

    def non_null_count(self) -> int:
        return sum(1 for v in self.model_dump().values() if v is not None)


class AssessRequest(BaseModel):
    """Request body for credit assessment."""

    features: dict[str, Any] = Field(
        ...,
        description="Applicant features matching Home Credit application columns",
        min_length=1,
    )
    model: Optional[str] = Field(
        default=None,
        description="Override model: xgboost | logistic_regression",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override decision threshold on P(default)",
    )
    include_explanation: bool = Field(
        default=True,
        description="If true and decision is decline, attach SHAP reasons",
    )
    top_k: int = Field(default=8, ge=1, le=30, description="Top SHAP features to return")

    @field_validator("features")
    @classmethod
    def _features_not_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v or not isinstance(v, dict):
            raise ValueError("features must be a non-empty object")
        # Reject if all values are null/None
        if all(val is None for val in v.values()):
            raise ValueError("features must include at least one non-null value")
        return v

    @field_validator("model")
    @classmethod
    def _model_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"xgboost", "logistic_regression"}
        if v not in allowed:
            raise ValueError(f"model must be one of {sorted(allowed)}")
        return v


class FeatureContribution(BaseModel):
    feature: str
    transformed_feature: Optional[str] = None
    shap_value: float
    contribution: float
    direction: str
    description: Optional[str] = None
    reason: str


class ExplanationBlock(BaseModel):
    model: str
    base_value: float
    top_features: list[FeatureContribution]
    decline_reasons: list[str]
    explanation_summary: str


class AssessResponse(BaseModel):
    decision: str
    default_probability: float
    risk_score: float
    model: str
    threshold: float
    message: str
    decline_reasons: list[str] = Field(default_factory=list)
    explanation: Optional[ExplanationBlock] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    production_model: Optional[str] = None
    threshold: Optional[float] = None
    n_features: Optional[int] = None
    metrics: Optional[dict[str, Any]] = None


class MetricsResponse(BaseModel):
    production_model: str
    threshold: float
    models: dict[str, Any]
    note: Optional[str] = None
