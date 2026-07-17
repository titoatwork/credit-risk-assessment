# Class imbalance & metrics FAQ (for skeptics)

## Q1. Is the data imbalanced?

**Yes.** Home Credit `TARGET=1` (payment difficulties) is about **8%** of applications; **~92%** are non-default. That is standard for credit default prediction.

---

## Q2. Does imbalance mean our model is invalid?

**No.** Imbalance means:

1. **Accuracy is a bad headline metric** (always-approve ≈ 92% accurate).
2. **Precision at a soft threshold can look modest** even for useful models.
3. We should report **ROC-AUC, PR-AUC, lift vs base rate, precision@top-k%**.

Our pipeline already uses:

- `class_weight="balanced"` (logistic regression)
- `scale_pos_weight ≈ n_neg/n_pos` (XGBoost)
- Stratified train/test splits
- Threshold selection (Youden / F1 / fixed)

---

## Q3. “Precision is only ~16% so the model is trash.”

**Incomplete reasoning.**

Precision is:

\[
\text{Precision} = \frac{\text{true defaults among those we flag}}{\text{everyone we flag}}
\]

If the population default rate is 8%, and we flag a large group to catch most defaulters (high recall), many non-defaulters enter the flagged set → precision falls.

**Better comparison:** *lift*

\[
\text{Lift} = \frac{\text{Precision}}{\text{Base rate}}
\]

Example: precision 16% / base rate 8% ⇒ **lift ≈ 2×**  
→ people we flag default **about twice as often** as a random applicant. That is skill, not noise.

Also look at **precision@top 10%**: among the 10% highest-risk scores, what fraction actually default? That is ranking quality without an arbitrary 0.5 cutoff.

---

## Q4. What numbers should convince a careful reviewer?

| Metric | Why it matters | Rough baseline |
|--------|----------------|----------------|
| **ROC-AUC** | Ranking quality | 0.50 random; ~0.70–0.76 application-only is common; multi-table often higher |
| **PR-AUC** | Ranking under imbalance | ~base rate (≈0.08) if no skill |
| **Lift@threshold** | How much riskier the flagged group is | 1.0 = no skill |
| **Precision@top 10%** | Quality of the riskiest tail | vs base rate |
| Precision/Recall@τ | Operating policy | Tradeoff; change τ to rebalance |

Kaggle Home Credit is scored on **ROC-AUC**, not precision.

---

## Q5. Will SMOTE / oversampling “fix” precision?

Often **not in the way people hope**.

- SMOTE can help some learners; evidence is mixed vs **class weights + threshold tuning**.
- You must **never** apply SMOTE to the test set.
- Tree models with `scale_pos_weight` often do fine without SMOTE.
- The largest real gains on Home Credit usually come from **multi-table feature engineering**, not resampling.

---

## Q6. How do we improve for real?

1. **Multi-table features** (`--multi-table`): bureau, previous apps, installments, credit card, POS  
2. **Full data** (`--sample-size 0`)  
3. **Richer derived features** (EXT_SOURCE combos, debt ratios) — implemented  
4. **Threshold policy** for the business goal (high precision vs high recall)  
5. Optional: calibration, hyperparameter search, LightGBM  

---

## Q7. One paragraph for your presentation

> The dataset is imbalanced (~8% defaults). We therefore optimize and report ranking metrics (ROC-AUC, PR-AUC) and lift relative to the base rate, not accuracy. Precision and recall are evaluated at a chosen decision threshold; they trade off and can be moved by changing τ. A precision near 16% with ~2× lift and ROC-AUC near 0.75 indicates useful risk ranking under rare events, consistent with application-level Home Credit baselines. Further gains come primarily from multi-table credit history features and full-data training.

---

## References (conceptual)

- Imbalanced classification metrics tours (ROC vs PR)
- Class weights vs SMOTE comparative studies (weights + threshold often competitive)
- Home Credit Kaggle: official metric **ROC-AUC**; strong solutions rely on extensive FE across related tables
