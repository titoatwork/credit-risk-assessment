"""Regenerate models/full_data/metrics_summary.md from metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "models" / "full_data" / "metrics.json"
m = json.loads(p.read_text(encoding="utf-8"))
lr, xgb = m["models"]["logistic_regression"], m["models"]["xgboost"]
prod = m["production_model"]
lines = [
    "# Model metrics summary (FULL DATA)",
    "",
    f"- Production model: **{prod}**",
    f"- Rows used: **{m['n_rows_used']}** (entire application_train)",
    f"- Multi-table: **{m['multi_table']}**",
    f"- Features: **{m['n_features']}**",
    f"- Test base rate: **{m['base_rate_test']:.2%}**",
    f"- Operating threshold (tau): **{m['threshold']:.4f}**",
    "",
    "| Model | ROC-AUC | PR-AUC | Precision@tau | Recall@tau | Lift@tau | Prec@top10% |",
    "|-------|---------|--------|---------------|------------|----------|-------------|",
    (
        f"| Logistic Regression | {lr['roc_auc']:.4f} | {lr['pr_auc']:.4f} | "
        f"{lr['precision']:.4f} | {lr['recall']:.4f} | "
        f"{lr['operating_point']['lift_vs_base_rate']:.2f}x | "
        f"{lr['ranking']['precision_at_top_10pct']:.4f} |"
    ),
    (
        f"| XGBoost | {xgb['roc_auc']:.4f} | {xgb['pr_auc']:.4f} | "
        f"{xgb['precision']:.4f} | {xgb['recall']:.4f} | "
        f"{xgb['operating_point']['lift_vs_base_rate']:.2f}x | "
        f"{xgb['ranking']['precision_at_top_10pct']:.4f} |"
    ),
    "",
    "## Note",
    "This artifact is separate from models/credit_risk_bundle.joblib (50k sample model).",
    "",
]
out = ROOT / "models" / "full_data" / "metrics_summary.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
