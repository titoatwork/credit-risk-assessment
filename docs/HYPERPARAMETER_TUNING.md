# Hyperparameter tuning (academic protocol)

## Integrity rule

| Data split | Used for tuning? |
|------------|------------------|
| **Train** | Fit each candidate |
| **Validation** | Select best hyperparameters + model + threshold |
| **Test** | **Never** used for tuning — final report only |

## How to run

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"

# Full data + hyperparameter search → models/full_data (separate from 50k pack)
.\.venv\Scripts\python.exe -m credit_risk.train `
  --sample-size 0 `
  --tune `
  --tune-iter 12 `
  --tune-mode holdout `
  --model auto `
  --threshold-strategy youden `
  --artifacts models/full_data/credit_risk_bundle.joblib `
  --metrics models/full_data/metrics.json
```

### Modes

| Flag | Meaning |
|------|---------|
| `--tune-mode holdout` | Random configs trained on train, ranked by **validation ROC-AUC** (default, laptop-friendly) |
| `--tune-mode cv` | `RandomizedSearchCV` + 3-fold stratified CV **on train only**, then checked on val |

### Search spaces

**Logistic regression**

- `C` ∈ {0.01 … 10}
- `solver` ∈ {lbfgs} (saga removed for speed/stability on this scale)
- `max_iter` ∈ {1000, 2000}
- Always `class_weight="balanced"`

**Status:** Full-data holdout tune **completed**; best XGB test ROC-AUC **0.769**. See `STATUS.md` and `models/full_data/metrics.json` → `hyperparameter_tuning`.

**XGBoost**

- `n_estimators`, `max_depth`, `learning_rate`
- `min_child_weight`, `subsample`, `colsample_bytree`
- `reg_lambda`, `gamma`
- `scale_pos_weight` near `n_neg/n_pos` (±50%)

## After search

1. Best LR and best XGB kept  
2. Production model chosen by **validation** ROC-AUC (PR-AUC tie-break)  
3. Threshold chosen on **validation**  
4. By default, models are **refit on train+val** with frozen hyperparameters  
5. **Test** metrics reported once  

Best parameters are stored in `metrics.json` → `hyperparameter_tuning`.

## Code

- `src/credit_risk/tuning.py` — search logic  
- `src/credit_risk/train.py --tune` — lifecycle integration  
