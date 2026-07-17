# Presentation guide (faculty / viva)

## One-sentence pitch

> “I built an end-to-end credit risk system on the full Home Credit train set (307k applications): stratified train/val/test, dummy baseline, hyperparameter-tuned logistic regression and XGBoost, validation-selected threshold, test ROC-AUC 0.77 for XGBoost, SHAP decline reasons, and a FastAPI + Streamlit interface.”

## Numbers to memorize

| Item | Value |
|------|--------|
| Rows | 307,511 |
| Dummy ROC-AUC | ~0.50 |
| LR test ROC-AUC | 0.75 |
| **XGB test ROC-AUC** | **0.769** (tuned) |
| Lift@τ (XGB) | ~1.9× |
| Base rate | ~8% |
| HP tuning | Yes — train/val only |

## Demo order (10 min)

1. **`STATUS.md` / `README.md`** — problem + lifecycle  
2. **`reports/eda/eda_report.md`** — imbalance & missingness  
3. **`models/full_data/metrics_summary.md`** — dummy vs LR vs XGB  
4. **Dashboard** — high-risk preset → decline + SHAP; low-risk → approve  
5. **API** `/docs` or `pytest -q`  

### Dashboard

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

### API

```powershell
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000
```

## Likely questions

**Why not accuracy?** ~92% non-default; always-approve is “accurate” but useless.  

**Why precision ~16%?** Base rate ~8%; at high recall, PPV is modest. Lift ~2× shows skill.  

**Train/val/test?** Models fit on train; model + τ on validation; metrics on test.  

**Full data?** Yes — primary pack is 307k, not only a sample.  

**Hyperparameter tuning?** Yes on full data: random search on train, select on validation, report on test. Best XGB test AUC 0.769.

## Files to open

| Priority | Path |
|----------|------|
| High | `STATUS.md` |
| High | `models/full_data/metrics_summary.md` |
| High | `docs/ML_LIFECYCLE.md` |
| High | `dashboard/app.py` (live) |
| Medium | `docs/IMBALANCE_AND_METRICS_FAQ.md` |
