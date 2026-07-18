# 5-Minute Technical Presentation Script  
## Credit Risk Assessment System

**Speaker:** Ibteshamul Haque  
**Repo:** https://github.com/titoatwork/credit-risk-assessment  
**Total time:** ~5:00 (tight; practice once with a timer)

---

## Before you start (30 seconds setup)

Have open on screen (tabs ready):

1. **Dashboard** — http://localhost:8501  
2. **GitHub repo** — README visible  
3. Optional: `models/full_data/metrics_summary.md`

Start dashboard beforehand:

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

**Memorize these 6 numbers:**

| Fact | Number |
|------|--------|
| Dataset size | **307,511** applicants |
| Default rate | **~8%** |
| Dummy ROC-AUC | **0.50** |
| XGBoost test ROC-AUC | **0.77** |
| Lift @ threshold | **~1.9×** |
| Production model | **XGBoost (tuned)** |

---

## Minute-by-minute script

### [0:00–0:40] Opening — problem & goal  
*(Speak clearly; face the audience. No slides required if dashboard is ready.)*

> “Good morning. I’m **Ibteshamul Haque**, and this is my project: a **Credit Risk Assessment System**.
>
> The problem is simple to state and hard to solve: given an applicant’s financial and demographic profile, should we **approve or decline** a credit product, and **why**?
>
> I use the **Home Credit Default Risk** dataset—**307,511** applications. The target is whether the client later had **payment difficulties**. Only about **8%** default, so this is a heavily **imbalanced** classification problem.
>
> My system does three things end-to-end: it **scores risk**, it makes an **approve/decline decision**, and for declines it explains the decision with **SHAP**—and all of this is packaged as a **dashboard and an API**.”

**On screen:** GitHub README (title + metrics table) *or* project folder.

---

### [0:40–1:30] ML lifecycle & methodology  
*(This is where you sound “senior academic.”)*

> “I followed a full supervised learning lifecycle—not just a notebook.
>
> **First, EDA:** class imbalance, missing values, numeric vs categorical features.  
> **Second, feature engineering:** domain ratios such as credit-to-income and annuity-to-income, external-source aggregates, and cleaning the employment-day anomaly used in Home Credit.  
> **Third, preprocessing inside a scikit-learn Pipeline:** median imputation for numerics, mode for categoricals, scaling for logistic regression, one-hot encoding with unknown-category handling so train and serve stay aligned.
>
> **Fourth, a proper split:** stratified **train / validation / test**—about 197k / 49k / 62k.  
> Models are fit on **train**. Hyperparameters, model choice, and the decision threshold are selected on **validation only**. **Test is frozen** for final reporting. That avoids optimistic leakage.
>
> I also include a **dummy stratified baseline**, so we can prove the models beat chance.”

**On screen (optional):** `docs/ML_LIFECYCLE.md` or speak only.

---

### [1:30–2:30] Models, imbalance, tuning, metrics  
*(Core technical meat. Slow down slightly.)*

> “I train two real models plus the baseline.
>
> **Logistic regression** with `class_weight=balanced`—interpretable baseline.  
> **XGBoost** with `scale_pos_weight` near the negative-to-positive ratio—strong tabular learner under imbalance.
>
> I ran **hyperparameter search** by fitting candidates on train and scoring ROC-AUC on validation—**never on test**. Best XGBoost used, among other settings, depth 6, learning rate 0.05, 350 trees, and tuned scale_pos_weight.
>
> After selection, I refit on train+val with frozen hyperparameters, then evaluate once on test.
>
> **Results on the held-out test set:**
>
> - Dummy ROC-AUC ≈ **0.50**—as expected for chance.  
> - Logistic regression ≈ **0.75**.  
> - **XGBoost ≈ 0.77**—selected as production.
>
> Primary ranking metric is **ROC-AUC**, which is also the Home Credit competition metric. I also report **PR-AUC**, because positives are rare.
>
> A common criticism is that precision looks low—around 15%. With an **8% base rate**, that is expected at a high-recall operating point. The right comparison is **lift**: at our threshold, flagged applicants default about **1.9 times** more often than average. The top 10% riskiest scores contain about **28%** true defaults—far above 8%. So the model concentrates risk; it is not random.”

**On screen:** `STATUS.md` or `models/full_data/metrics_summary.md` table.

---

### [2:30–3:40] Live demo — dashboard (the “wow”)  
*(Click slowly; narrate what they see.)*

> “Now the system in production form—a Streamlit dashboard loading the full-data XGBoost pack.”

**Actions:**

1. Show sidebar: **artifact pack `models/full_data`**, production **XGBoost**, rows **307511**.  
2. Load preset **High risk profile** → **Run assessment**.  
3. Point to: **DECLINE**, probability gauge, threshold, **SHAP bar chart**, decline reasons.  

> “For a high-risk profile the model **declines**. The gauge shows predicted default probability. SHAP attributes the prediction to features—typically external credit scores and credit burden—so we can say *why* risk is high, not only *that* it is high.”

4. Load **Low risk profile** → **Run assessment**.  
5. Point to: **APPROVE**, lower probability.  

> “A safer profile is **approved**. Same pipeline, same threshold policy—different risk score.”

**Backup if dashboard fails:**  
`python scripts/example_assess.py` or GitHub README metrics only—do not panic.

---

### [3:40–4:30] System architecture & engineering  

> “Beyond modeling, this is software.
>
> The core package separates **data, preprocess, train, predict, explain, and API**. Training writes a joblib bundle with both models, the threshold, feature schema, and metadata. The API—**FastAPI**—exposes health, metrics, assess, and explain endpoints with request validation. The dashboard is a thin UI over the same predict and SHAP path.
>
> I maintain automated tests for train, predict, explain, API, evaluation, and tuning. The project is on GitHub with documentation for lifecycle, metrics FAQ, model card, and this presentation guide.
>
> Large model binaries are kept local and gitignored; metrics and code are versioned so the repository stays lightweight and reproducible.”

**On screen:** GitHub repo structure (folders: `src`, `docs`, `tests`, `dashboard`).

---

### [4:30–5:00] Closing — limitations & takeaway  

> “Limitations, briefly: multi-table credit history is implemented as an optional path but not the default serve model; real banks would add fairness monitoring, calibration for pricing, and regulatory adverse-action language beyond SHAP. This is an **educational end-to-end system**, not a production underwriting engine.
>
> **Takeaway:** I built a complete credit-risk pipeline on the full Home Credit train set—from EDA and imbalance-aware training, through validation-based tuning and thresholding, to explainable decisions served in a dashboard and API—with transparent metrics that beat a dummy baseline and concentrate default risk about two-fold.
>
> Thank you. I’m happy to take questions.”

**Stop. Smile. Hands free.**

---

## Demo click path (cheat sheet)

| Time | Click |
|------|--------|
| 2:35 | Dashboard open, sidebar visible |
| 2:45 | Preset → **High risk** → Run assessment |
| 3:10 | Point to SHAP / decline reasons |
| 3:20 | Preset → **Low risk** → Run assessment |
| 3:35 | Optional: Model metrics tab (ROC-AUC bars) |

---

## 30-second “emergency” version  
*(If cut short)*

> “Credit risk system on 307k Home Credit applicants. Imbalanced ~8% defaults. Stratified train/val/test; dummy baseline 0.50 AUC; production XGBoost 0.77 AUC after validation tuning. Decision by threshold on default probability; declines explained with SHAP. Served as FastAPI and Streamlit. Code on GitHub under titoatwork/credit-risk-assessment.”

---

## Anticipated Q&A (prepare, don’t read)

### “Why is precision so low?”
> “Base rate is 8%. At high recall, precision is mathematically capped. Our lift is about 1.9× and top-decile precision about 28%. Ranking skill is ROC-AUC 0.77 vs 0.50 dummy.”

### “Why not accuracy?”
> “Always-approve is ~92% accurate and useless. Accuracy hides failure on the rare class.”

### “Did you leak the test set?”
> “No. Hyperparameters, model choice, and threshold use validation only. Test is reported once.”

### “Why XGBoost over logistic regression?”
> “Both trained; XGBoost won on validation ROC-AUC and kept the lead on test (~0.77 vs ~0.75). LR remains as an interpretable challenger in the bundle.”

### “Is SHAP causal?”
> “No—local feature attribution for transparency, not causal proof. Good for demo and internal explanation; legal adverse action needs policy mapping.”

### “What’s next?”
> “Full multi-table features in the serve path, probability calibration, and cost-sensitive thresholds.”

---

## Delivery tips

1. **Speak slower than you think**—technical talks feel fast to listeners.  
2. **One idea per sentence.**  
3. When demoing, **pause 1 second** after each click so the screen updates.  
4. If interrupted, jump to the **30-second emergency** version.  
5. Never apologize for precision without immediately saying **lift + AUC**.  

---

## Timing card (print this)

```text
0:00  Problem + goal
0:40  Lifecycle + split integrity
1:30  Models, tuning, metrics, imbalance
2:30  LIVE DEMO high → low risk
3:40  Engineering: API, tests, GitHub
4:30  Limits + close
5:00  STOP
```

**Repo to mention once:**  
`github.com/titoatwork/credit-risk-assessment`
