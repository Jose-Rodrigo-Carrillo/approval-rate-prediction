# Learning Log

Personal record of concepts studied, resources used, and insights gained
during this project. Honest, incremental, written for future reference.

---

## Week 1 — Project setup and data understanding

### What I did
- Set up repo structure: `src/`, `notebooks/`, `data/`, `reports/`
- Read UCI dataset documentation and literature on the double-header
  issue (`skiprows=1`)
- Wrote `data.py`: load, clean, recode undocumented categories, create target

### What I learned
- **Data leakage is subtle**: fitting the scaler on the full dataset before
  splitting is a common mistake that produces optimistic metrics. The rule is
  simple — the scaler must only see training data.
- **Undocumented categories matter**: EDUCATION values 0, 5, 6 are not in
  the original codebook. Leaving them as-is implies ordinal meaning that
  does not exist. Always verify value ranges against the data dictionary.
- **`engine="xlrd"` is required for `.xls`**: openpyxl silently fails on
  legacy Excel format. Small detail, real bug.

### Resources
- UCI ML Repository: Credit Card Default dataset description
- pandas `read_excel` documentation

---

## Week 2 — Feature engineering

### What I did
- Wrote `features.py` with 19 engineered features
- Validated each feature group against the target using correlation analysis
- Ran EDA notebook: distributions, correlations, repayment status breakdown

### What I learned
- **Domain knowledge directly improves AUC**: translating "bill / limit"
  into a utilisation ratio (a concept I know from payments) added +0.04 AUC
  over raw features alone. Feature engineering is where domain expertise
  becomes a quantifiable advantage.
- **Vectorised OLS slope**: computing a trend across 6 time points for 30k
  rows using matrix operations (`bill_matrix @ x`) is orders of magnitude
  faster than a row-wise loop. NumPy broadcasting is worth learning deeply.
- **Negative BILL_AMT is not an error**: values below zero represent credits
  or refunds. Clipping to 0 before log transform is the correct handling —
  not dropping or flagging as bad data.
- **Log1p vs log**: `np.log1p(x)` handles x=0 correctly (returns 0).
  `np.log(0)` returns -inf. Always use log1p for financial amounts.

### Resources
- Hands-On Machine Learning (Géron) — Chapter 2: feature engineering
- NumPy broadcasting documentation

---

## Week 3 — Modelling and evaluation

### What I did
- Trained Logistic Regression baseline and Random Forest
- Built evaluation framework: ROC-AUC, Precision, Recall, F1
- Ran threshold sweep analysis
- Compared feature importances across both models

### What I learned
- **AUC gap is the overfitting signal**: LogReg gap ~0.00, RF gap ~0.10.
  The RF is more powerful but memorises more. `min_samples_leaf=20` helps
  but does not eliminate the gap — hyperparameter tuning is the next step.
- **Threshold 0.5 is arbitrary**: in payments, the business defines the
  operating point. A lower threshold approves more clients (higher recall,
  lower precision). The threshold sweep chart makes this trade-off explicit
  and is the most business-relevant output of this project so far.
- **Precision-Recall curve > ROC curve under imbalance**: ROC can look
  optimistic when the negative class dominates. PR curve shows the real
  cost of increasing recall.
- **F1 maximisation is not always the right objective**: in fraud detection
  or risk, a business might prefer high precision (low false approvals)
  even at the cost of recall. Understanding the cost asymmetry matters more
  than picking the "best" metric.
- **class_weight="balanced" changes training dynamics**: without it, the
  model ignores the minority class. With it, each defaulted client has
  ~3.5x the weight of an approved client during training.

### Resources
- scikit-learn documentation: LogisticRegression, RandomForestClassifier
- "Calibration of probabilities" — sklearn user guide
- Kaggle notebook: UCI Credit Card — benchmark AUC references

---

## Ongoing — Concepts to deepen

- [ ] Cross-validation: why `StratifiedKFold` and not `KFold` under imbalance
- [ ] Hyperparameter tuning: `RandomizedSearchCV` vs `GridSearchCV` trade-offs
- [ ] SHAP values: model-agnostic feature explanation
- [ ] Calibration: are the predicted probabilities well-calibrated?
- [ ] MLflow: experiment tracking for comparing runs systematically
- [ ] pytest: writing unit tests for `build_features()`