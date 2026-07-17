"""
Multi-table feature engineering for Home Credit.

Aggregates bureau / previous applications / installments / balances to SK_ID_CURR
and joins onto application-level features.

Laptop-safe: when ``id_filter`` is provided (recommended after sampling
application_train), large tables are read in chunks and only matching IDs kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

from credit_risk.config import DATA_DIR, ID_COL


def _id_set(ids: Optional[Iterable]) -> Optional[Set[int]]:
    if ids is None:
        return None
    out: Set[int] = set()
    for x in ids:
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            continue
    return out or None


def _read_csv_filtered(
    path: Path,
    id_filter: Optional[Set[int]] = None,
    usecols: Optional[list[str]] = None,
    chunksize: int = 400_000,
) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if id_filter is None:
        return pd.read_csv(path, usecols=usecols)

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        if ID_COL not in chunk.columns:
            return None
        # IDs may be float in some dumps
        mask = chunk[ID_COL].astype("int64", errors="ignore").isin(id_filter)
        sub = chunk.loc[mask]
        if len(sub):
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=usecols or [ID_COL])
    return pd.concat(parts, ignore_index=True)


def aggregate_bureau(
    data_dir: Path = DATA_DIR,
    id_filter: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    path = data_dir / "bureau.csv"
    prefer = [
        ID_COL,
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "CREDIT_CURRENCY",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_TYPE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
    ]
    df = _read_csv_filtered(path, id_filter=id_filter, usecols=lambda c: c in prefer)
    if df is None or df.empty:
        return None

    num_cols = [
        c
        for c in df.columns
        if c not in (ID_COL, "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    aggs = {c: ["mean", "max", "sum", "min"] for c in num_cols}
    grouped = df.groupby(ID_COL).agg(aggs)
    grouped.columns = [f"BUREAU_{a}_{b}".upper() for a, b in grouped.columns]
    grouped["BUREAU_COUNT"] = df.groupby(ID_COL).size()
    if "CREDIT_ACTIVE" in df.columns:
        grouped["BUREAU_ACTIVE_COUNT"] = (
            df.assign(_a=(df["CREDIT_ACTIVE"] == "Active").astype(int))
            .groupby(ID_COL)["_a"]
            .sum()
        )
        grouped["BUREAU_CLOSED_COUNT"] = (
            df.assign(_c=(df["CREDIT_ACTIVE"] == "Closed").astype(int))
            .groupby(ID_COL)["_c"]
            .sum()
        )
    if "AMT_CREDIT_SUM_OVERDUE" in df.columns:
        grouped["BUREAU_HAS_OVERDUE"] = (
            df.groupby(ID_COL)["AMT_CREDIT_SUM_OVERDUE"].max().gt(0).astype(int)
        )
    return grouped.reset_index()


def aggregate_previous_application(
    data_dir: Path = DATA_DIR,
    id_filter: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    path = data_dir / "previous_application.csv"
    prefer = [
        ID_COL,
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
        "NFLAG_INSURED_ON_APPROVAL",
    ]
    df = _read_csv_filtered(path, id_filter=id_filter, usecols=lambda c: c in prefer)
    if df is None or df.empty:
        return None

    num_cols = [
        c
        for c in df.columns
        if c not in (ID_COL, "SK_ID_PREV", "NAME_CONTRACT_STATUS")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    aggs = {c: ["mean", "max", "sum", "min"] for c in num_cols[:30]}
    grouped = df.groupby(ID_COL).agg(aggs)
    grouped.columns = [f"PREV_{a}_{b}".upper() for a, b in grouped.columns]
    grouped["PREV_APP_COUNT"] = df.groupby(ID_COL).size()
    if "NAME_CONTRACT_STATUS" in df.columns:
        for status, name in [
            ("Approved", "PREV_APPROVAL_RATE"),
            ("Refused", "PREV_REFUSED_RATE"),
            ("Canceled", "PREV_CANCELED_RATE"),
        ]:
            grouped[name] = (
                df.assign(_s=(df["NAME_CONTRACT_STATUS"] == status).astype(int))
                .groupby(ID_COL)["_s"]
                .mean()
            )
    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(df.columns):
        df = df.copy()
        df["_credit_app_ratio"] = df["AMT_CREDIT"] / df["AMT_APPLICATION"].replace(0, np.nan)
        grouped["PREV_CREDIT_APP_RATIO_MEAN"] = df.groupby(ID_COL)["_credit_app_ratio"].mean()
    return grouped.reset_index()


def aggregate_installments(
    data_dir: Path = DATA_DIR,
    id_filter: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    path = data_dir / "installments_payments.csv"
    usecols = [ID_COL, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    df = _read_csv_filtered(path, id_filter=id_filter, usecols=lambda c: c in usecols)
    if df is None or df.empty:
        return None

    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(df.columns):
        df = df.copy()
        df["PAYMENT_DELAY"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["LATE_PAYMENT"] = (df["PAYMENT_DELAY"] > 0).astype(int)
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(df.columns):
        df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)
        df["UNDERPAID"] = (df["AMT_PAYMENT"] < df["AMT_INSTALMENT"]).astype(int)

    aggs: dict[str, list[str]] = {}
    for c in ("PAYMENT_DELAY", "PAYMENT_RATIO", "AMT_PAYMENT", "AMT_INSTALMENT", "LATE_PAYMENT", "UNDERPAID"):
        if c in df.columns:
            aggs[c] = ["mean", "max", "sum"]
    grouped = df.groupby(ID_COL).agg(aggs)
    grouped.columns = [f"INST_{a}_{b}".upper() for a, b in grouped.columns]
    grouped["INST_COUNT"] = df.groupby(ID_COL).size()
    return grouped.reset_index()


def aggregate_credit_card_balance(
    data_dir: Path = DATA_DIR,
    id_filter: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    path = data_dir / "credit_card_balance.csv"
    prefer = [
        ID_COL,
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_RECEIVABLE_PRINCIPAL",
        "AMT_RECIVABLE",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    df = _read_csv_filtered(path, id_filter=id_filter, usecols=lambda c: c in prefer)
    if df is None or df.empty:
        return None
    df = df.copy()
    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(df.columns):
        df["CC_UTILIZATION"] = df["AMT_BALANCE"] / df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    num_cols = [c for c in df.columns if c != ID_COL and pd.api.types.is_numeric_dtype(df[c])]
    aggs = {c: ["mean", "max", "sum"] for c in num_cols}
    grouped = df.groupby(ID_COL).agg(aggs)
    grouped.columns = [f"CC_{a}_{b}".upper() for a, b in grouped.columns]
    return grouped.reset_index()


def aggregate_pos_cash(
    data_dir: Path = DATA_DIR,
    id_filter: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    path = data_dir / "POS_CASH_balance.csv"
    prefer = [ID_COL, "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF", "NAME_CONTRACT_STATUS"]
    df = _read_csv_filtered(path, id_filter=id_filter, usecols=lambda c: c in prefer)
    if df is None or df.empty:
        return None
    num_cols = [
        c
        for c in df.columns
        if c not in (ID_COL, "NAME_CONTRACT_STATUS") and pd.api.types.is_numeric_dtype(df[c])
    ]
    aggs = {c: ["mean", "max"] for c in num_cols}
    grouped = df.groupby(ID_COL).agg(aggs)
    grouped.columns = [f"POS_{a}_{b}".upper() for a, b in grouped.columns]
    grouped["POS_COUNT"] = df.groupby(ID_COL).size()
    return grouped.reset_index()


def build_multitable_features(
    application: pd.DataFrame,
    data_dir: Optional[Path] = None,
    include: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Left-join multi-table aggregates onto application-level frame.

    Filters large tables to SK_ID_CURR present in ``application`` to save RAM.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    include = include or ["bureau", "previous", "installments", "credit_card", "pos_cash"]
    out = application.copy()
    id_filter = _id_set(out[ID_COL].tolist()) if ID_COL in out.columns else None

    builders = {
        "bureau": aggregate_bureau,
        "previous": aggregate_previous_application,
        "installments": aggregate_installments,
        "credit_card": aggregate_credit_card_balance,
        "pos_cash": aggregate_pos_cash,
    }
    for key in include:
        fn = builders.get(key)
        if fn is None:
            continue
        agg = fn(data_dir, id_filter=id_filter)
        if agg is None or agg.empty:
            continue
        out = out.merge(agg, on=ID_COL, how="left")

    return out


def add_application_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain ratios and risk-relevant transforms used in strong Home Credit baselines."""
    out = df.copy()

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"].replace(0, np.nan)
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(out.columns):
        out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"].replace(0, np.nan)
        out["CREDIT_GOODS_DIFF"] = out["AMT_CREDIT"] - out["AMT_GOODS_PRICE"]
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["INCOME_PER_PERSON"] = out["AMT_INCOME_TOTAL"] / out["CNT_FAM_MEMBERS"].replace(0, np.nan)
    if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(out.columns):
        out["INCOME_PER_CHILD"] = out["AMT_INCOME_TOTAL"] / (out["CNT_CHILDREN"] + 1)

    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = (-out["DAYS_BIRTH"]) / 365.25
    if "DAYS_EMPLOYED" in out.columns:
        # Home Credit uses 365243 as missing / pensioner sentinel — do not scale as real days
        anomaly = out["DAYS_EMPLOYED"] == 365243
        out["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype(int)
        out["DAYS_EMPLOYED_CLEAN"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
        out["YEARS_EMPLOYED"] = (-out["DAYS_EMPLOYED_CLEAN"]) / 365.25
        # Replace raw column so LR/XGB never see 365243 as a huge real value
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED_CLEAN"]
        if "AGE_YEARS" in out.columns:
            out["EMPLOYMENT_TO_AGE_RATIO"] = out["YEARS_EMPLOYED"] / out["AGE_YEARS"].replace(0, np.nan)

    # External source combinations (often among strongest signals)
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in out.columns]
    if ext_cols:
        out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
        out["EXT_SOURCE_MIN"] = out[ext_cols].min(axis=1)
        out["EXT_SOURCE_MAX"] = out[ext_cols].max(axis=1)
        out["EXT_SOURCE_PROD"] = out[ext_cols].prod(axis=1, min_count=1)
        out["EXT_SOURCE_MISSING_COUNT"] = out[ext_cols].isna().sum(axis=1)
        if len(ext_cols) >= 2:
            out["EXT_SOURCE_STD"] = out[ext_cols].std(axis=1)

    # Document flags sum if present
    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        out["FLAG_DOCUMENT_SUM"] = out[doc_cols].sum(axis=1)

    # Contact flags
    contact_cols = [
        c
        for c in (
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        )
        if c in out.columns
    ]
    if contact_cols:
        out["FLAG_CONTACT_SUM"] = out[contact_cols].sum(axis=1)

    return out
