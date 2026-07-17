"""
Run the full academic ML lifecycle end-to-end on full Home Credit data.

Outputs:
  - reports/eda/*
  - models/full_data/credit_risk_bundle.joblib  (does not overwrite models/*.joblib sample pack)
  - models/full_data/metrics.json
  - models/full_data/metrics_summary.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.train import train_models  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    out_dir = ROOT / "models" / "full_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_models(
        sample_size=None,  # entire application_train
        artifacts_path=out_dir / "credit_risk_bundle.joblib",
        metrics_path=out_dir / "metrics.json",
        threshold_strategy="youden",
        preferred_model="auto",
        multi_table=False,
        run_eda=True,
    )
    print(
        json.dumps(
            {
                "lifecycle": "academic_v2_train_val_test",
                "production_model": result.production_model,
                "threshold": result.threshold,
                "artifacts_path": result.artifacts_path,
                "metrics_path": result.metrics_path,
                "n_models_reported": len(result.metrics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
