# Model metrics summary (FULL DATA + HYPERPARAMETER TUNING — PRIMARY)

- Production model: **xgboost**
- Rows used: **307511** (entire application_train)
- Lifecycle: **academic_v2_train_val_test**
- Hyperparameter tuning: **yes** (holdout random search on train/val; test unused)
- Split: train **196806** / val **49202** / test **61503**
- Threshold selected on: **validation** (tau=0.2691, youden)
- Final metrics on: **test**
- Multi-table: **False** | Features: **140**
- Test base rate: **8.07%**

| Model | Split | ROC-AUC | PR-AUC | Precision@tau | Recall@tau | Lift@tau | Prec@top10% |
|-------|-------|---------|--------|---------------|------------|----------|-------------|
| Dummy (stratified) | test | 0.5016 | 0.0810 | 0.0837 | 0.0828 | 1.04x | 0.0818 |
| Logistic Regression | test | 0.7525 | 0.2364 | 0.1033 | 0.9305 | 1.28x | 0.2680 |
| **XGBoost** | test | **0.7690** | **0.2610** | 0.1537 | 0.7619 | **1.90x** | **0.2839** |

## Best hyperparameters (validation-selected)

### XGBoost (production)
- n_estimators=350, max_depth=6, learning_rate=0.05
- min_child_weight=5, subsample=0.7, colsample_bytree=1.0
- reg_lambda=0.5, gamma=0.0, scale_pos_weight≈5.69
- Best validation ROC-AUC during search: **0.7637**

### Logistic regression
- C=0.5, solver=saga, max_iter=500
- Best validation ROC-AUC during search: **0.7482**

## Academic checklist

- [x] Problem definition
- [x] EDA report
- [x] Feature engineering + Pipeline preprocessing
- [x] Stratified train/validation/test
- [x] Baseline dummy model
- [x] LR + XGBoost with imbalance handling
- [x] **Hyperparameter tuning (train/val only)**
- [x] Model selection on validation
- [x] Threshold selection on validation
- [x] Refit on train+val with frozen hyperparams
- [x] Final metrics on held-out test
- [x] SHAP + API + dashboard

## Note

Default serve pack for API/dashboard. Sample 50k model remains at `models/credit_risk_bundle.joblib`.
