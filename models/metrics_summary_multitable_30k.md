# Model metrics summary

- Production model: **logistic_regression**
- Sample size: **30000** (rows used: 30000)
- Multi-table features: **True**
- Features: **312**
- Test base rate (default %): **8.07%**
- Operating threshold τ: **0.4527** (youden)

| Model | ROC-AUC | PR-AUC | Precision@τ | Recall@τ | Lift@τ | Prec@top10% |
|-------|---------|--------|-------------|----------|--------|-------------|
| Logistic Regression | 0.7592 | 0.2236 | 0.1657 | 0.7397 | 2.05x | 0.2633 |
| XGBoost | 0.7516 | 0.2282 | 0.2312 | 0.3802 | 2.87x | 0.2617 |

## How to read this under imbalance

- Random baseline default rate ≈ **8.1%**.
- Lift@τ > 1 means declined/flagged group defaults more often than average (production lift **2.05x**).
- Precision@top10% = share of true defaults among the 10% highest-risk scores (threshold-free view of ranking quality on the tail).
- See `metrics.json` → `threshold_sweep` for precision/recall at many cutoffs.
