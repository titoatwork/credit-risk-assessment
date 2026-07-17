# Architecture

## Components

| Module | Responsibility |
|--------|----------------|
| `config.py` | Paths, defaults (threshold, sample size, model preference) |
| `data.py` | Load `application_train.csv`, stratified sampling, column descriptions |
| `preprocess.py` | Numeric/categorical pipelines (impute, scale, one-hot) |
| `features_multitable.py` | Derived ratios; optional bureau/previous/installment aggregates |
| `train.py` | Fit LR + XGBoost, evaluate, select threshold, save artifacts |
| `predict.py` | Load artifacts; `P(default)` → approve/decline |
| `explain.py` | SHAP local explanations for a single applicant |
| `schemas.py` | Pydantic validation for HTTP I/O |
| `api.py` | FastAPI routes and lifecycle (load models on startup) |

## Training flow

```text
load_application_train
        │
        ▼
[optional] build_multitable_features
        │
        ▼
add_application_derived_features
        │
        ▼
split_features_target  →  train_test_split (stratified)
        │
        ├─► Pipeline(preprocess, LogisticRegression).fit
        └─► Pipeline(preprocess, XGBClassifier).fit
        │
        ▼
compare ROC-AUC → production_model
        │
        ▼
select_threshold (youden | f1 | fixed)
        │
        ▼
joblib.dump(bundle) + metrics.json
```

## Inference flow

```text
HTTP POST /assess  { features, threshold?, model?, include_explanation? }
        │
        ▼
validate (Pydantic)
        │
        ▼
recompute derived features
        │
        ▼
align columns → Pipeline.predict_proba
        │
        ▼
decision = decline if p >= τ else approve
        │
        ▼
if decline or include_explanation:
        SHAP top-k features → decline_reasons
        │
        ▼
JSON response
```

## Artifact bundle contents

**Primary path:** `models/full_data/credit_risk_bundle.joblib` (full 307k academic model).  
Fallback / comparison: `models/credit_risk_bundle.joblib` (50k sample).

The joblib bundle includes:

- `logistic_regression`, `xgboost` — fitted pipelines  
- `production_model`, `threshold`, `threshold_info`  
- `feature_columns`, `transformed_feature_names`  
- `column_descriptions` — human-readable labels for SHAP text  
- `background_X` — sample for LinearExplainer (LR)  
- `metrics` — evaluation snapshot  
- `use_derived_features`, `multi_table`, `sample_size`

## Design principles

1. **Single pipeline object** for preprocess + model (train/serve parity)  
2. **No training inside the API** — API only loads artifacts  
3. **Imbalance-aware** training and evaluation  
4. **Explanations only as valuable as the model** — SHAP is local attribution, not causal proof  
5. **Configurable sample size** for reproducible demos on limited hardware  
