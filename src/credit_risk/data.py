"""Load Home Credit application data and optional column descriptions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from credit_risk.config import (
    APPLICATION_TRAIN,
    COLUMNS_DESCRIPTION,
    ID_COL,
    RANDOM_STATE,
    TARGET_COL,
)


def load_application_train(
    path: Optional[Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Load application_train.csv.

    Parameters
    ----------
    path : Path, optional
        CSV path (defaults to configured Home Credit path).
    sample_size : int, optional
        If set and positive, stratified sample of this many rows (for fast train/tests).
    random_state : int
        RNG seed for sampling.
    """
    csv_path = Path(path) if path is not None else APPLICATION_TRAIN
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Application train file not found: {csv_path}. "
            "Set CREDIT_RISK_DATA_DIR or place application_train.csv there."
        )

    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected column {TARGET_COL!r} in {csv_path}")

    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        # Stratified sample to preserve class balance
        parts = []
        for _, group in df.groupby(TARGET_COL, group_keys=False):
            n = max(1, int(round(sample_size * len(group) / len(df))))
            n = min(n, len(group))
            parts.append(group.sample(n=n, random_state=random_state))
        df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        # Trim if overshoot due to rounding
        if len(df) > sample_size:
            df = df.iloc[:sample_size].reset_index(drop=True)

    return df


def load_column_descriptions(path: Optional[Path] = None) -> dict[str, str]:
    """
    Map column name -> human-readable description from HomeCredit_columns_description.csv.
    Returns empty dict if file missing.
    """
    desc_path = Path(path) if path is not None else COLUMNS_DESCRIPTION
    if not desc_path.exists():
        return {}

    # File may be latin-1 encoded
    try:
        raw = pd.read_csv(desc_path, encoding="utf-8")
    except UnicodeDecodeError:
        raw = pd.read_csv(desc_path, encoding="latin-1")

    # Columns vary slightly; common: Row, Table, Row, Description or similar
    col_name = None
    desc_name = None
    for c in raw.columns:
        cl = c.lower().strip()
        if cl in ("row", "column", "column_name", "feature"):
            col_name = c
        if "description" in cl:
            desc_name = c
    if col_name is None or desc_name is None:
        # Try positional: often Table, Row, Description
        if len(raw.columns) >= 3:
            col_name = raw.columns[1]
            desc_name = raw.columns[2]
        else:
            return {}

    mapping: dict[str, str] = {}
    for _, row in raw.iterrows():
        key = str(row[col_name]).strip()
        val = str(row[desc_name]).strip()
        if key and key != "nan" and val and val != "nan":
            mapping[key] = val
    return mapping


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop ID and TARGET; return X, y."""
    drop_cols = [c for c in (TARGET_COL, ID_COL) if c in df.columns]
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=drop_cols)
    return X, y
