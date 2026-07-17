# Academic machine learning lifecycle

This project follows a **standard supervised learning lifecycle** suitable for coursework and viva defense.

```text
1. Problem definition
2. Data acquisition
3. Exploratory data analysis (EDA)
4. Data preparation & feature engineering
5. Train / validation / test split (stratified)
6. Preprocessing Pipeline (fit on train only)
7. Baseline model(s)
8. Candidate models (LR, XGBoost)
9. Model selection on VALIDATION
10. Threshold selection on VALIDATION
11. Final evaluation on UNTOUCHED TEST
12. Explainability (SHAP)
13. Deployment (API + dashboard)
14. Documentation & limitations
```

## Stage mapping to code

| Stage | Implementation |
|-------|----------------|
| Problem | Credit default risk → approve/decline (`docs/product.md`) |
| Data | `data.load_application_train` — Home Credit `application_train.csv` |
| EDA | `python -m credit_risk.eda` → `reports/eda/` |
| Features | `features_multitable.add_application_derived_features` (+ optional multi-table) |
| Split | Stratified **train / val / test** in `train.train_models` |
| Preprocess | `preprocess.build_preprocessor` inside sklearn `Pipeline` |
| Baseline | `DummyClassifier` (stratified) metrics on test |
| Models | Logistic Regression (`class_weight=balanced`), XGBoost (`scale_pos_weight`) |
| Hyperparameter tuning | `--tune` on full data **completed** for primary pack; search on train, select on **validation** (never test). See `docs/HYPERPARAMETER_TUNING.md` |
| Selection | Production model chosen by **validation** ROC-AUC (PR-AUC tie-break) |
| Threshold | Youden / F1 / fixed on **validation only** |
| Final metrics | Ranking + operating metrics on **test only** |
| Explain | `explain.explain_applicant` (SHAP) |
| Serve | FastAPI + Streamlit; artifacts in `models/` or `models/full_data/` |

## What was wrong before (and fixed)

| Issue | Academic problem | Fix |
|-------|------------------|-----|
| Single train/test split | Threshold tuned on same fold as reported metrics → optimistic P/R | **Train / val / test**; τ on val, report on test |
| No dummy baseline | Cannot prove models beat chance | **DummyClassifier** baseline on test |
| No formal EDA artifact | Lifecycle incomplete for reports | **`reports/eda/`** generated in lifecycle |
| Accuracy-centric peers | Misleading under 8% positives | ROC-AUC, PR-AUC, lift, precision@top-k |

## How to run the full lifecycle

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"

# EDA only
.\.venv\Scripts\python.exe -m credit_risk.eda --sample-size 0

# Full academic train on ALL rows → separate full_data artifacts
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 0 --model auto --threshold-strategy youden --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json

# Or one-shot lifecycle script
.\.venv\Scripts\python.exe scripts/run_academic_lifecycle.py
```

## Integrity rules (exam-ready)

1. **No target leakage** — TARGET never used as a feature.  
2. **Fit preprocess on train only** — Pipeline ensures val/test transform only.  
3. **Stratify** all splits on TARGET.  
4. **Test set is frozen** for final numbers after model + τ chosen.  
5. Disclose sample size vs full data and limitations (multi-table optional).  
