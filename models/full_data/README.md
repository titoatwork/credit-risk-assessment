# Full-data academic model (PRIMARY — tuned)

Trained on **all** Home Credit `application_train.csv` rows (**307,511**) with **completed hyperparameter tuning**.

| Field | Value |
|-------|--------|
| Lifecycle | `academic_v2_train_val_test` |
| Hyperparameter tuning | **Completed** (holdout; test never used) |
| Train / val / test | 196,806 / 49,202 / 61,503 |
| Production | **xgboost** |
| Test ROC-AUC | **0.7690** |
| Test PR-AUC | **0.2610** |
| Threshold τ | **0.2691** (Youden on validation) |
| Dummy baseline AUC | 0.5016 |

Best XGB params: n_estimators=350, max_depth=6, learning_rate=0.05, min_child_weight=5, subsample=0.7, colsample_bytree=1.0, reg_lambda=0.5, scale_pos_weight≈5.69.

## Default for serve

`credit_risk.config` loads this pack when the joblib exists.

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000
```

## Retrain with tuning

```powershell
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 0 --tune --tune-iter 10 --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json --no-eda
```
