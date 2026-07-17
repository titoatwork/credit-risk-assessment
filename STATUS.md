# Project status (source of truth)

**Last synced:** after full-data hyperparameter tuning (completed)

## Primary production model

| Field | Value |
|-------|--------|
| **Pack** | `models/full_data/` |
| **Rows** | **307,511** (entire `application_train.csv`) |
| **Lifecycle** | Academic train / validation / test |
| **Hyperparameter tuning** | **Completed** (holdout random search; TEST never used) |
| **Production algorithm** | **XGBoost** |
| **Best XGB val ROC-AUC (search)** | **0.7637** |
| **Test ROC-AUC (XGB)** | **0.7690** |
| **Test PR-AUC (XGB)** | **0.2610** |
| **Threshold τ** | **0.2691** (Youden on validation) |
| **Lift@τ (XGB)** | **1.90×** |
| **Prec@top10% (XGB)** | **0.2839** |
| **Dummy baseline ROC-AUC** | **0.5016** |

### Best hyperparameters (from search)

**XGBoost (production)**

| Param | Value |
|-------|--------|
| n_estimators | 350 |
| max_depth | 6 |
| learning_rate | 0.05 |
| min_child_weight | 5 |
| subsample | 0.7 |
| colsample_bytree | 1.0 |
| reg_lambda | 0.5 |
| gamma | 0.0 |
| scale_pos_weight | ≈ 5.69 |

**Logistic regression (challenger)**

| Param | Value |
|-------|--------|
| C | 0.5 |
| solver | saga* |
| max_iter | 500 |

\*Search included saga; future searches prefer `lbfgs` only (faster/stable). Metrics still valid.

### Full test comparison

| Model | Test ROC-AUC | Test PR-AUC | Precision@τ | Recall@τ | Lift@τ |
|-------|--------------|-------------|-------------|----------|--------|
| Dummy | 0.502 | 0.081 | 0.084 | 0.083 | 1.04× |
| Logistic Regression | 0.752 | 0.236 | 0.103 | 0.931 | 1.28× |
| **XGBoost** | **0.769** | **0.261** | 0.154 | 0.762 | **1.90×** |

Split: train **196,806** / val **49,202** / test **61,503**.

## Academic lifecycle checklist

```text
[x] Problem definition
[x] Data load (full 307k)
[x] EDA → reports/eda/
[x] Feature engineering
[x] Preprocess Pipeline (fit train only)
[x] Stratified train/val/test
[x] Dummy baseline
[x] Logistic Regression + XGBoost
[x] Hyperparameter tuning (train fit, val select)
[x] Model selection on validation
[x] Threshold on validation
[x] Refit train+val with frozen hyperparams
[x] Final metrics on test
[x] SHAP + API + Dashboard
[x] Docs + tests
```

## Comparison packs

| Pack | Rows | Role |
|------|------|------|
| `models/full_data/` | 307,511 | **PRIMARY** (tuned) |
| `models/` | 50,000 | Sample comparison |
| `*multitable_30k*` | 30,000 | Multi-table research |

## Commands

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"

# Dashboard / API (auto full_data)
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000

# Tests
.\.venv\Scripts\python.exe -m pytest -q

# Retrain full + tune again
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 0 --tune --tune-iter 10 --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json --no-eda
```
