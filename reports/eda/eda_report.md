# EDA Report — Home Credit application data

## Dataset size
- Rows: **307,511**
- Columns: **122** (modeling features: **120**)

## Target distribution (class imbalance)
- Default rate (TARGET=1): **8.07%**
- Class counts: `{'0': 282686, '1': 24825}`
- Neg/pos ratio: **11.39**

## Feature types
- Numeric: **104**
- Categorical: **16**

## Highest missingness (top features)

| Feature | Missing rate |
|---------|--------------|
| `COMMONAREA_AVG` | 69.9% |
| `COMMONAREA_MODE` | 69.9% |
| `COMMONAREA_MEDI` | 69.9% |
| `NONLIVINGAPARTMENTS_AVG` | 69.4% |
| `NONLIVINGAPARTMENTS_MODE` | 69.4% |
| `NONLIVINGAPARTMENTS_MEDI` | 69.4% |
| `FONDKAPREMONT_MODE` | 68.4% |
| `LIVINGAPARTMENTS_AVG` | 68.4% |
| `LIVINGAPARTMENTS_MEDI` | 68.4% |
| `LIVINGAPARTMENTS_MODE` | 68.4% |
| `FLOORSMIN_MODE` | 67.8% |
| `FLOORSMIN_AVG` | 67.8% |
| `FLOORSMIN_MEDI` | 67.8% |
| `YEARS_BUILD_AVG` | 66.5% |
| `YEARS_BUILD_MODE` | 66.5% |

## Implications for the ML pipeline

1. Use **stratified** splits to preserve class balance.
2. Report **ROC-AUC / PR-AUC**, not accuracy alone.
3. Apply **median/mode imputation** and **one-hot encoding** inside a Pipeline.
4. Handle imbalance with **class_weight / scale_pos_weight** and threshold policy.

_JSON report: `C:\Users\Ibteshamul Haque\credit-risk-assessment\reports\eda\eda_report.json`_
