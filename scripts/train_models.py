"""CLI entry: python scripts/train_models.py [--sample-size N]."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.train import main

if __name__ == "__main__":
    main()
