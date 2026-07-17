"""
Exploratory Data Analysis (EDA) for the academic ML lifecycle.

Produces a markdown + JSON report: shape, class balance, missingness,
numeric summaries, and categorical cardinalities — without mutating data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from credit_risk.config import PROJECT_ROOT, TARGET_COL
from credit_risk.data import load_application_train, split_features_target


def run_eda(
    sample_size: Optional[int] = None,
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Run EDA on application_train and write reports under reports/eda/.

    Returns the report dict (also written to eda_report.json).
    """
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "reports" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_application_train(path=data_path, sample_size=sample_size)
    n_rows, n_cols = df.shape
    target_rate = float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else None
    class_counts = (
        df[TARGET_COL].value_counts().sort_index().to_dict()
        if TARGET_COL in df.columns
        else {}
    )
    # Convert numpy types for JSON
    class_counts = {str(k): int(v) for k, v in class_counts.items()}

    X, y = split_features_target(df)
    missing = X.isna().mean().sort_values(ascending=False)
    missing_top = {str(k): float(v) for k, v in missing.head(25).items() if v > 0}

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    num_summary = {}
    if numeric_cols:
        desc = X[numeric_cols].describe().T
        # keep a compact subset of high-missing or core fields if huge
        for col in list(desc.index)[:40]:
            row = desc.loc[col]
            num_summary[str(col)] = {
                "mean": float(row["mean"]) if pd.notna(row["mean"]) else None,
                "std": float(row["std"]) if pd.notna(row["std"]) else None,
                "min": float(row["min"]) if pd.notna(row["min"]) else None,
                "max": float(row["max"]) if pd.notna(row["max"]) else None,
            }

    cat_summary = {}
    for col in cat_cols[:30]:
        cat_summary[str(col)] = {
            "n_unique": int(X[col].nunique(dropna=True)),
            "top": str(X[col].mode(dropna=True).iloc[0]) if X[col].notna().any() else None,
            "missing_rate": float(X[col].isna().mean()),
        }

    report: dict[str, Any] = {
        "stage": "1_exploratory_data_analysis",
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "n_features_modeling": int(X.shape[1]),
        "target_column": TARGET_COL,
        "class_counts": class_counts,
        "default_rate": target_rate,
        "imbalance_ratio_neg_pos": (
            float((y == 0).sum() / max(int((y == 1).sum()), 1)) if len(y) else None
        ),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(cat_cols),
        "top_missing_rates": missing_top,
        "numeric_summary_sample": num_summary,
        "categorical_summary_sample": cat_summary,
        "academic_notes": [
            "Class imbalance (~8% positives) implies accuracy is a poor primary metric.",
            "Missing values are common in Home Credit; imputation is required before modeling.",
            "Categorical fields require encoding (one-hot) with handle_unknown for serve-time safety.",
        ],
    }

    json_path = output_dir / "eda_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# EDA Report — Home Credit application data",
        "",
        "## Dataset size",
        f"- Rows: **{n_rows:,}**",
        f"- Columns: **{n_cols}** (modeling features: **{X.shape[1]}**)",
        "",
        "## Target distribution (class imbalance)",
        f"- Default rate (TARGET=1): **{(target_rate or 0):.2%}**",
        f"- Class counts: `{class_counts}`",
        f"- Neg/pos ratio: **{report['imbalance_ratio_neg_pos']:.2f}**",
        "",
        "## Feature types",
        f"- Numeric: **{len(numeric_cols)}**",
        f"- Categorical: **{len(cat_cols)}**",
        "",
        "## Highest missingness (top features)",
        "",
    ]
    if missing_top:
        md_lines.append("| Feature | Missing rate |")
        md_lines.append("|---------|--------------|")
        for k, v in list(missing_top.items())[:15]:
            md_lines.append(f"| `{k}` | {v:.1%} |")
    else:
        md_lines.append("_No missing values in sampled frame._")

    md_lines.extend(
        [
            "",
            "## Implications for the ML pipeline",
            "",
            "1. Use **stratified** splits to preserve class balance.",
            "2. Report **ROC-AUC / PR-AUC**, not accuracy alone.",
            "3. Apply **median/mode imputation** and **one-hot encoding** inside a Pipeline.",
            "4. Handle imbalance with **class_weight / scale_pos_weight** and threshold policy.",
            "",
            f"_JSON report: `{json_path}`_",
            "",
        ]
    )
    md_path = output_dir / "eda_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    report["paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def main(argv: Optional[list] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EDA for Home Credit application data")
    parser.add_argument("--sample-size", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)
    sample = None if args.sample_size == 0 else args.sample_size
    report = run_eda(
        sample_size=sample,
        data_path=Path(args.data_path) if args.data_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps({"n_rows": report["n_rows"], "default_rate": report["default_rate"], "paths": report.get("paths")}, indent=2))


if __name__ == "__main__":
    main()
