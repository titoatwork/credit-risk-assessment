"""Export a real application_train row as API JSON for demos/tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import APPLICATION_TRAIN, ID_COL, TARGET_COL
from credit_risk.data import load_application_train


def main() -> None:
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_application_train(sample_size=500, random_state=42)
    # One default + one non-default if available
    samples = []
    for target in (1, 0):
        subset = df[df[TARGET_COL] == target]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]
        feats = {
            k: (None if (isinstance(v, float) and v != v) else v)
            for k, v in row.drop(labels=[TARGET_COL], errors="ignore").items()
            if k != ID_COL
        }
        # Drop pure NaNs for a cleaner payload
        feats = {k: v for k, v in feats.items() if v is not None}
        samples.append(
            {
                "label_target": int(target),
                "label_meaning": "default_risk" if target == 1 else "no_default",
                "features": feats,
            }
        )

    path = out_dir / "sample_payloads.json"
    path.write_text(json.dumps(samples, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {path} ({len(samples)} samples) from {APPLICATION_TRAIN}")


if __name__ == "__main__":
    main()
