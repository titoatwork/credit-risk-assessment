# Model card — Credit Risk Assessment

## Primary model (production)

| Field | Value |
|-------|--------|
| **Path** | `models/full_data/credit_risk_bundle.joblib` |
| **Lifecycle** | `academic_v2_train_val_test` + hyperparameter tuning |
| **Production algorithm** | **XGBoost** |
| **Rows** | **307,511** |
| **Test ROC-AUC** | **0.769** |
| **Test PR-AUC** | **0.261** |
| **Threshold τ** | **0.269** (Youden on validation) |
| **Intended use** | Educational credit risk approve/decline + SHAP |

## Training data

Home Credit `application_train.csv`, TARGET = payment difficulties (~8.07% positive).  
140 features (application + derived).

## Split protocol

| Split | Size | Role |
|-------|------|------|
| Train | 196,806 | Fit + HP search candidates |
| Validation | 49,202 | HP selection, model choice, threshold |
| Test | 61,503 | Final metrics only |

## Hyperparameter tuning

- Mode: holdout random search (`--tune --tune-mode holdout --tune-iter 10`)
- Fit on train; score on validation; **test never used**
- Best XGB val ROC-AUC: **0.7637** → test **0.7690**
- Best XGB params: n_estimators=350, max_depth=6, lr=0.05, min_child_weight=5, subsample=0.7, colsample_bytree=1.0, reg_lambda=0.5, scale_pos_weight≈5.69

## Test metrics

| Model | ROC-AUC | PR-AUC | Lift@τ | Prec@top10% |
|-------|---------|--------|--------|-------------|
| Dummy | 0.502 | 0.081 | 1.04× | 0.082 |
| Logistic Regression | 0.752 | 0.236 | 1.28× | 0.268 |
| **XGBoost** | **0.769** | **0.261** | **1.90×** | **0.284** |

## Decision policy

Decline if \(P(\text{default}) \ge \tau\).

## Explainability

SHAP TreeExplainer (XGB) / LinearExplainer (LR).

## Reproducibility

```powershell
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 0 --tune --tune-iter 10 --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json
.\.venv\Scripts\python.exe -m pytest -q
```

Seed: `RANDOM_STATE = 42`.
