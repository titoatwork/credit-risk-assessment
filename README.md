# Credit Risk Assessment System

**Academic end-to-end ML project** — predict default risk from Home Credit application data, decide **approve / decline**, explain declines with **SHAP**, serve via **FastAPI** + **Streamlit dashboard**.

| | |
|--|--|
| **Author** | Ibteshamul Haque |
| **Primary model pack** | `models/full_data/` (**307,511** rows) |
| **Production model** | **XGBoost** (test ROC-AUC **0.769**, hyperparameter-tuned) |
| **Lifecycle** | Train / val / test + dummy baseline + HP tuning |
| **Stack** | scikit-learn · XGBoost · SHAP · FastAPI · Streamlit |

Live status: [`STATUS.md`](STATUS.md)

---

## 1. Problem

Home Credit `TARGET = 1` means **payment difficulties** (~8% of applicants).  
We map predicted default probability to a credit-card-style decision:

| Rule | Decision |
|------|----------|
| \(P(\text{default}) \ge \tau\) | **decline** |
| \(P(\text{default}) < \tau\) | **approve** |

Threshold \(\tau\) is chosen on the **validation** set (Youden); reported metrics use the **test** set only.

---

## 2. Academic ML lifecycle

```text
Problem → Data → EDA → Features → Stratified train/val/test
  → Pipeline preprocess (fit train only)
  → Dummy baseline + Logistic Regression + XGBoost
  → Select model & threshold on VALIDATION
  → Final metrics on TEST
  → SHAP explanations → API / Dashboard
```

Details: [`docs/ML_LIFECYCLE.md`](docs/ML_LIFECYCLE.md)

---

## 3. Current results (full data + tuning — primary)

**Source:** `models/full_data/metrics.json`  
**Split:** train 196,806 · val 49,202 · test 61,503 · base rate ≈ 8.07%  
**HP tuning:** completed (random search on train/val; test never used)  
**τ ≈ 0.269** (Youden on validation after tuning)

| Model | Test ROC-AUC | Test PR-AUC | Precision@τ | Recall@τ | Lift@τ | Prec@top10% |
|-------|--------------|-------------|-------------|----------|--------|-------------|
| Dummy (stratified) | 0.502 | 0.081 | 0.084 | 0.083 | 1.04× | 0.082 |
| Logistic Regression | 0.752 | 0.236 | 0.103 | 0.931 | 1.28× | 0.268 |
| **XGBoost (production)** | **0.769** | **0.261** | 0.154 | 0.762 | **1.90×** | **0.284** |

Best XGB (val): n_estimators=350, max_depth=6, learning_rate=0.05, min_child_weight=5, subsample=0.7, scale_pos_weight≈5.69.

Under ~8% defaults, precision alone looks low; **lift ≈ 1.9×** and **ROC-AUC 0.77 vs dummy 0.50** show real skill.  
See [`docs/IMBALANCE_AND_METRICS_FAQ.md`](docs/IMBALANCE_AND_METRICS_FAQ.md) · [`STATUS.md`](STATUS.md).

### Other packs (comparison only)

| Pack | Rows | Note |
|------|------|------|
| `models/` | 50k sample | Older LR model (~0.749 AUC) — not default serve |
| `*multitable_30k*` | 30k + history tables | Research ablation (~0.759 LR) |

---

## 4. Repository layout

```text
credit-risk-assessment/
├── STATUS.md                 ← current numbers & checklist
├── README.md
├── requirements.txt
├── docs/                     ← lifecycle, tuning, FAQ, presentation
├── reports/eda/              ← EDA report
├── models/
│   ├── full_data/            ← PRIMARY full-data academic model
│   └── *.joblib / metrics*   ← sample & multitable comparisons
├── src/credit_risk/          ← package (train, predict, explain, api, tuning, eda)
├── dashboard/app.py          ← Streamlit UI
├── scripts/
└── tests/
```

---

## 5. Setup

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Dataset default:

`C:\Users\Ibteshamul Haque\Downloads\home-credit-default-risk\application_train.csv`

---

## 6. Run dashboard (enter values & score)

Uses **full-data** model automatically when present:

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open **http://localhost:8501** → Score applicant → presets or your values → Run assessment.

Force sample model:

```powershell
$env:CREDIT_RISK_MODELS_DIR = "$PWD\models"
```

---

## 7. Run API

```powershell
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000
```

Docs: http://127.0.0.1:8000/docs  
Endpoints: `/health` · `/metrics` · `/features` · `/assess` · `/explain`

---

## 8. Train

```powershell
# Full academic lifecycle (307k) → models/full_data/
.\.venv\Scripts\python.exe scripts\run_academic_lifecycle.py

# Fast iteration sample
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 30000 --model auto --no-eda

# Optional multi-table features
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 30000 --multi-table --model auto

# Re-run full data + hyperparameter tuning (already completed once)
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 0 --tune --tune-iter 10 --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json --no-eda
```

Tuning docs: [`docs/HYPERPARAMETER_TUNING.md`](docs/HYPERPARAMETER_TUNING.md) — **primary pack already includes a completed tune run.**

---

## 9. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## 10. Presentation

Use [`docs/PRESENTATION_GUIDE.md`](docs/PRESENTATION_GUIDE.md).

**One-liner for faculty:**

> Full Home Credit train (307k), stratified train/val/test, dummy baseline ROC-AUC ≈ 0.50, production XGBoost test ROC-AUC ≈ 0.77, threshold on validation, SHAP decline reasons, FastAPI + Streamlit.

---

## 11. License & data

MIT (see `LICENSE`). Home Credit data under Kaggle terms — educational use; not a production bank system.
