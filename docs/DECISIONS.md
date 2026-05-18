# Architecture & Design Decisions

This document records key technical decisions made during the project,
the reasoning behind them, and the alternatives considered.

---

## DD-001 — Dataset choice: UCI Credit Card Default

**Date:** 2025  
**Status:** Accepted

### Decision
Use the UCI Credit Card Default dataset (Taiwan, 2005) as the primary
dataset for this project.

### Reasoning
- Publicly available, no licensing restrictions — safe for a public portfolio
- Domain relevance: default prediction maps directly to approval rate monitoring
  in a payments context (the closer the real-world problem, the more transferable
  the learnings)
- Well-studied benchmark: external AUC references exist (~0.77–0.82), making
  it easy to validate whether the pipeline is correct

### Alternatives considered
- Synthetic dataset generated from scratch: more control, but less credible
  as a portfolio piece
- Kaggle credit datasets: many have data quality issues or unclear provenance

---

## DD-002 — Three-way split: train / val / test (70 / 15 / 15)

**Date:** 2025  
**Status:** Accepted

### Decision
Split data into three sets: train (70%), validation (15%), test (15%),
with stratification on the target.

### Reasoning
- A dedicated test set that is never used during development ensures
  the final reported metrics are unbiased estimates of real-world performance
- Validation set allows model selection and threshold tuning without
  touching the test set
- Stratification preserves the 78/22 class ratio in every split —
  without it, a small split could end up with a different imbalance ratio

### Alternatives considered
- Cross-validation only (no hold-out test): common but harder to communicate
  a single final number to a non-technical audience
- 80/10/10 split: with 30,000 rows, 15% test = 4,500 samples, sufficient
  for stable metric estimates

---

## DD-003 — StandardScaler fitted on train only

**Date:** 2025  
**Status:** Accepted

### Decision
Fit `StandardScaler` exclusively on `X_train`, then apply `.transform()`
to val and test.

### Reasoning
- Fitting the scaler on val or test would leak distribution information
  from those sets into the training process — a form of data leakage
- Even a small leakage can produce optimistically biased metrics that
  do not reflect production performance
- This is enforced in `features.scale_features()` by design, not convention

### Alternatives considered
- `MinMaxScaler`: sensitive to outliers; `BILL_AMT` and `PAY_AMT` have
  extreme values (max ~873k), so MinMax would compress most values near 0
- No scaling for Random Forest: RF is scale-invariant, but scaling
  does not hurt and keeps the pipeline consistent across models

---

## DD-004 — Recoding undocumented EDUCATION and MARRIAGE values

**Date:** 2025  
**Status:** Accepted

### Decision
Recode `EDUCATION` values {0, 4, 5, 6} and `MARRIAGE` value {0} to
category 3 ("Other"), as documented in `data.py`.

### Reasoning
- The original UCI codebook defines EDUCATION as 1=graduate, 2=university,
  3=high school, 4=others — but the actual data contains values 0, 5, and 6
  with no explanation
- Leaving them as separate numeric values would imply an ordinal relationship
  that does not exist, misleading the model
- Merging into "Other" (3) is the standard approach in the literature for
  this dataset

### Alternatives considered
- Drop rows with undocumented values: would lose ~3% of the dataset with
  no clear benefit
- One-hot encode all categories: valid, but adds columns for categories
  that have no documented meaning

---

## DD-005 — class_weight="balanced" on all models

**Date:** 2025  
**Status:** Accepted

### Decision
Use `class_weight="balanced"` in both Logistic Regression and Random Forest.

### Reasoning
- The dataset has a 78/22 class imbalance (approved vs defaulted)
- Without weighting, both models would be biased toward always predicting
  "approved", achieving 78% accuracy while having poor recall on defaults
- `balanced` automatically sets weights inversely proportional to class
  frequency: `n_samples / (n_classes * np.bincount(y))`

### Alternatives considered
- SMOTE oversampling: adds synthetic minority samples; more complex and
  harder to justify in a baseline notebook
- Threshold tuning alone: can achieve similar recall adjustment post-hoc,
  but does not affect training — combining both is optimal

---

## DD-006 — Feature engineering: 19 new features from raw columns

**Date:** 2025  
**Status:** Accepted

### Decision
Engineer 19 features in `features.build_features()` across four groups:
utilisation ratios, repayment ratios, delay behaviour aggregates, and
log transforms.

### Reasoning
- Raw columns encode information implicitly: BILL_AMT alone is less
  informative than BILL_AMT / LIMIT_BAL (utilisation)
- PAY_RATIO captures how much of the bill was actually repaid — not
  available directly in the raw data
- Log transforms on LIMIT_BAL and payment amounts reduce the impact of
  extreme values on linear models
- BILL_TREND (OLS slope across 6 months) captures whether debt is growing
  or shrinking — a time-series signal from cross-sectional data

### Impact
LogReg AUC with only raw features: ~0.73  
LogReg AUC after feature engineering: ~0.772 (+0.04)

---

## DD-007 — Random Forest hyperparameters: n_estimators=300, min_samples_leaf=20

**Date:** 2025  
**Status:** Accepted

### Decision
Use 300 trees with a minimum of 20 samples per leaf as the initial
Random Forest configuration.

### Reasoning
- `n_estimators=300`: enough trees for variance stabilisation on 21k
  training rows; beyond ~500 trees, gains are marginal
- `min_samples_leaf=20`: acts as implicit regularisation — each leaf
  must represent at least 20 clients, preventing the model from
  memorising individual noise patterns
- Result: train-test AUC gap reduced to ~0.10 vs ~0.15+ with default params

### Alternatives considered
- Default sklearn RF (n_estimators=100, min_samples_leaf=1): significant
  overfitting observed (train AUC ~0.99)
- XGBoost: stronger model but more hyperparameters; reserved for next iteration