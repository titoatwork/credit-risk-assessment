# Credit Risk Assessment — Design Notes & Research Recommendations

Companion reading for the implementation in `src/credit_risk/`.  
For demos and viva structure, prefer [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md).

## Problem framing

You want a system that decides whether a person should get a **credit card** given **credit history**, with:

1. **Logistic regression** (interpretable baseline)
2. **XGBoost** (strong tabular model)
3. **SHAP** explanations when credit is **declined**
4. An **API service** for scoring

Your dataset is **Home Credit Default Risk** (Kaggle). Important nuance:

| Dataset reality | Product mapping |
|-----------------|-----------------|
| `TARGET = 1` = client had **payment difficulties** | High `P(default)` → **decline** card |
| `TARGET = 0` = repaid without difficulty (as labeled) | Low `P(default)` → **approve** |
| Not literally “card issuer approved/denied” labels | Treat as **risk-based underwriting** |

This is standard industry practice: score default risk, apply a policy threshold.

---

## Why these two models

### Logistic regression
- Strong baseline for credit risk / scorecards
- Coefficients map cleanly to “which factors raise/lower risk”
- Works well with standardized numeric features + one-hot categoricals
- Use `class_weight="balanced"` because defaults are ~8% of rows

### XGBoost
- Usually best-in-class on tabular credit data with mixed types
- Handles non-linearities and interactions
- Use `scale_pos_weight ≈ n_neg/n_pos` for imbalance
- Prefer **ROC-AUC** and **PR-AUC** over accuracy (accuracy is misleading on imbalance)

### Production choice
- Train **both**, pick production by validation **ROC-AUC** (or PR-AUC if defaults are rare and ranking positives matters more).
- On a 20k sample of `application_train`, LR and XGB were close (~0.69 AUC); full multi-table features usually favor XGBoost.

---

## Metrics that matter (not accuracy)

| Metric | Why |
|--------|-----|
| **ROC-AUC** | Ranking quality: can the model put higher risk first? |
| **PR-AUC** | Better when positives (defaults) are rare |
| Precision @ threshold | Of declined (or flagged high-risk), how many truly default? |
| Recall @ threshold | Of true defaulters, how many do we catch? |
| Calibration | Does 20% predicted risk ≈ 20% observed defaults? |

**Do not** report accuracy alone on Home Credit.

---

## Threshold policy (business decision)

Default in code: `0.5`. That is **not** sacred.

Better approaches:
1. Sweep thresholds on validation; pick by **F1**, **F2** (recall-heavy), or cost matrix  
2. Cap portfolio risk: e.g. approve only if predicted PD < 5%  
3. Cap decline rate: e.g. decline worst 20% of scores  

Document the chosen policy in `docs/product.md` for interviews/portfolio.

---

## SHAP for “why declined”

- **XGBoost**: `shap.TreeExplainer` (fast, standard)
- **Logistic regression**: `shap.LinearExplainer` or coefficient × centered features
- Return top-k features with **direction**: increased vs decreased risk
- Map raw columns via `HomeCredit_columns_description.csv` for human-readable reasons

**Compliance note:** Real adverse-action notices (ECOA/FCRA) need carefully worded reason codes. SHAP is excellent for **transparency / demos / internal tools**; legal letters need product/legal review.

---

## Feature engineering roadmap (full system)

### Phase A (current MVP path) — `application_train.csv`
~120 application-time features: income, credit amount, external sources, demographics, housing, etc.  
`EXT_SOURCE_1/2/3` are typically among the strongest predictors.

### Phase B (recommended for “fully fledged”)
Aggregate other tables to `SK_ID_CURR` then join:

| Table | Example aggregates |
|-------|--------------------|
| `bureau` | count of bureaus, active credits, sum/mean debt, max overdue |
| `bureau_balance` | months balance stats per bureau → roll up |
| `previous_application` | # previous apps, approval rate, mean annuity/credit |
| `installments_payments` | payment delay stats, fraction paid late |
| `POS_CASH_balance` | DPD stats, active contracts |
| `credit_card_balance` | utilization, drawings, DPD |

Code stub lives in `src/credit_risk/features_multitable.py` for incremental expansion.

### Phase C (advanced)
- Target encoding (carefully, with CV)
- Interaction features (credit/income ratio, annuity/income)
- Time-based stability checks
- Fairness audit by age/gender bands

---

## API design

| Endpoint | Role |
|----------|------|
| `GET /health` | Model loaded? production name, threshold |
| `GET /metrics` | Train/eval metrics JSON |
| `GET /features` | Expected feature columns |
| `POST /assess` | Decision + PD; SHAP on decline |
| `POST /explain` | Always attach SHAP explanation |

Invalid payloads → **422** (validation) / **400** (scoring failure) / **503** (no artifacts).

---

## Training safely on a laptop (research paper season)

- **CPU-only** stack — no GPU required for this project  
- Use `--sample-size N` while iterating  
- Full train (`--sample-size 0`) when you can leave the PC idle  
- Keep research env and this project **separate venv**  
- Avoid simultaneous huge downloads + full train + games  

Hardware crash risk (e.g. display driver BSOD) is **orthogonal** to model code; don’t update GPU drivers mid-paper if that risks your other work.

---

## Suggested portfolio narrative

1. Problem: risk-based credit card approve/decline  
2. Data: Home Credit, imbalance, multi-table potential  
3. Models: LR baseline + XGBoost challenger  
4. Metrics: ROC-AUC / PR-AUC, not accuracy  
5. Explainability: SHAP decline reasons via API  
6. Engineering: reproducible train → artifacts → FastAPI  
7. Future: multi-table FE, threshold optimization, fairness, monitoring  

---

## References (starting points)

- Home Credit Default Risk competition discussions (feature ideas)  
- Scikit-learn pipelines & imbalanced classification guides  
- XGBoost docs: `scale_pos_weight`  
- SHAP papers/docs: TreeExplainer for gradient boosting  
- Credit risk: PD/LGD/EAD framing (here we model **PD** only)
