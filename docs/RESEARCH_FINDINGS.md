# Deep research findings — Credit Risk Assessment

*Synthesized from codebase audit, public Home Credit / imbalance literature, and local experiments (2026).*

---

## 1. What the official metric is

The Home Credit Default Risk competition (and almost all serious credit-default work) scores models on **ROC-AUC**, not accuracy or raw precision. Top public solutions historically sit near **~0.80 ROC-AUC** with heavy multi-table feature engineering. Application-only models commonly land around **0.70–0.76**.

---

## 2. Realistic performance bands

| Setup | Typical ROC-AUC |
|-------|-----------------|
| Random | 0.50 |
| Application only, simple models | 0.70–0.74 |
| Application + domain ratios + solid GBM/LR | **0.75–0.76** |
| + bureau / previous_application | 0.77–0.78 |
| Full multi-table FE | 0.78–0.80 |
| Top ensembles | ~0.80–0.805 |

**Our measured results (this repo):**

| Experiment | Rows | Features | LR ROC-AUC | XGB ROC-AUC | Notes |
|------------|------|----------|------------|-------------|--------|
| Sample pack `models/` | 50k | ~140 | **0.749** | 0.743 | Comparison only |
| Multi-table research | 30k | ~312 | **0.759** | 0.752 | Ablation |
| **PRIMARY full data + HP tune** `models/full_data/` | **307,511** | 140 | 0.752 | **0.769** | Academic train/val/test + tuning; production **XGB** |

Authoritative numbers: `STATUS.md` and `models/full_data/metrics_summary.md`.

---

## 3. Why precision ~16% is not “the model is random”

With base rate \(\pi \approx 0.08\):

\[
\text{Precision} = \frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi + \text{FPR}\cdot(1-\pi)}
\]

Example: recall 70%, specificity 70% (FPR 30%) ⇒ precision ≈ **17%** and lift ≈ **2.1×**.

Our extended metrics report (app-only upgrade):

- **Lift@operating threshold ≈ 2.0×** (flagged group defaults ~2× population rate)
- **Precision@top 10% ≈ 0.25–0.27** (top risk decile is much denser in defaults than 8%)

So the model has ranking skill; low precision at a high-recall threshold is **base-rate math**.

---

## 4. Imbalance techniques — what research recommends here

| Technique | Verdict for this project |
|-----------|---------------------------|
| Class weights / `scale_pos_weight` | **Use** (already) |
| Threshold optimization | **Use** (Youden/F1; report full sweep) |
| PR-AUC + lift + precision@top-k | **Report** (now in `metrics.json`) |
| SMOTE on XGBoost | **Usually skip** — little AUC gain, hurts calibration |
| Multi-table FE | **Highest real AUC lift** |
| Full data train | Stability + modest lift |

Literature comparing SMOTE vs class weights vs threshold adjustment finds threshold/class-weight approaches often competitive; SMOTE is not a free lunch for tree models.

---

## 5. Codebase audit — main risks fixed or remaining

| Issue | Status |
|-------|--------|
| Thin derived features | **Improved** (EXT combos, goods ratio, etc.) |
| DAYS_EMPLOYED = 365243 distorting LR | **Fixed** (replace with NaN + anomaly flag) |
| Multi-table incomplete / not memory-safe | **Improved** (chunked ID filter + POS_CASH) |
| No lift / threshold sweep in metrics | **Added** (`evaluation.py`) |
| Multi-table train/serve skew if API lacks history features | **Documented** — production serve model remains app-level; multi-table artifacts saved separately for research comparison |
| Threshold tuned on same holdout as metrics | Known academic limitation; OK for class project if disclosed |

---

## 6. How to speak to skeptics (short)

1. Base rate is 8% — precision alone is prevalence-dependent.  
2. Report lift and ROC-AUC (Kaggle metric).  
3. 16% precision ≈ 2× base rate at high recall is skill.  
4. We can raise precision by raising τ (with lower recall).  
5. Real upgrades = multi-table history + full data, not only “balance the CSV with SMOTE.”

Full FAQ: [`IMBALANCE_AND_METRICS_FAQ.md`](IMBALANCE_AND_METRICS_FAQ.md).

---

## 7. Commands used for experiments

```powershell
# App-only, richer FE
python -m credit_risk.train --sample-size 50000 --model auto --threshold-strategy youden

# Multi-table research run
python -m credit_risk.train --sample-size 30000 --multi-table --model auto --threshold-strategy youden
```

Artifacts:

- `models/metrics.json` — current production (serve-safe)
- `models/metrics_multitable_30k.json` — multi-table comparison
- `models/metrics_summary.md` — human-readable table
