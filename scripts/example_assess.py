"""Example client: POST /assess against a running API (or score in-process)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Representative high-risk-ish payload (raw Home Credit-style fields)
SAMPLE_FEATURES = {
    "AMT_INCOME_TOTAL": 90000,
    "AMT_CREDIT": 540000,
    "AMT_ANNUITY": 28000,
    "AMT_GOODS_PRICE": 540000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -800,
    "CNT_CHILDREN": 1,
    "EXT_SOURCE_1": 0.25,
    "EXT_SOURCE_2": 0.30,
    "EXT_SOURCE_3": 0.20,
    "CODE_GENDER": "M",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_INCOME_TYPE": "Working",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_CONTRACT_TYPE": "Cash loans",
}


def in_process() -> dict:
    from credit_risk.predict import assess_applicant, load_bundle, reset_bundle_cache
    from credit_risk.explain import explain_decline

    reset_bundle_cache()
    bundle = load_bundle()
    assessment = assess_applicant(SAMPLE_FEATURES, bundle=bundle)
    return explain_decline(SAMPLE_FEATURES, assessment, bundle=bundle)


def via_http(base_url: str) -> dict:
    import httpx

    payload = {
        "features": SAMPLE_FEATURES,
        "include_explanation": True,
        "top_k": 8,
    }
    r = httpx.post(f"{base_url.rstrip('/')}/assess", json=payload, timeout=60.0)
    r.raise_for_status()
    return r.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=None,
        help="API base URL e.g. http://127.0.0.1:8000 (default: score in-process)",
    )
    args = parser.parse_args()
    out = via_http(args.url) if args.url else in_process()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
