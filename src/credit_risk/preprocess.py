"""Feature preprocessing pipeline shared by training and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols)."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer:
    - numeric: median impute (+ optional standard scale)
    - categorical: most_frequent impute + one-hot encode
    """
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(steps=num_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Feature names after fit (requires sklearn >= 1.0)."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def prepare_input_frame(
    features: dict[str, Any],
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to training columns.
    Missing keys become NaN; unknown keys are ignored.
    """
    row = {col: features.get(col, np.nan) for col in expected_columns}
    return pd.DataFrame([row], columns=expected_columns)
