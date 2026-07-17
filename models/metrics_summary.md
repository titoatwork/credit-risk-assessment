# Model metrics summary

- Production model: **logistic_regression**
- Sample size: **50000** (rows used: 50000)
- Multi-table features: **False**
- Features: **140**
- Test base rate (default %): **8.07%**
- Operating threshold τ: **0.5031** (youden)

| Model | ROC-AUC | PR-AUC | Precision@τ | Recall@τ | Lift@τ | Prec@top10% |
|-------|---------|--------|-------------|----------|--------|-------------|
| Logistic Regression | 0.7493 | 0.2117 | 0.1639 | 0.6840 | 2.03x | 0.2570 |
| XGBoost | 0.7430 | 0.2356 | 0.2110 | 0.4808 | 2.61x | 0.2570 |

## How to read this under imbalance

- Random baseline default rate ≈ **8.1%**.
- Lift@τ > 1 means declined/flagged group defaults more often than average (production lift **2.03x**).
- Precision@top10% = share of true defaults among the 10% highest-risk scores (threshold-free view of ranking quality on the tail).
- See `metrics.json` → `threshold_sweep` for precision/recall at many cutoffs.
