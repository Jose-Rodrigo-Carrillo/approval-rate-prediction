# Project Progress

High-level tracker of what is done, in progress, and planned.
Updated at the end of each working session.

---

## Status Overview

| Phase | Status | Notes |
|---|---|---|
| 1. Repo setup | ✅ Done | Structure, README, .gitignore |
| 2. Data pipeline | ✅ Done | `data.py`: load, clean, split |
| 3. Feature engineering | ✅ Done | `features.py`: 19 features |
| 4. Baseline model | ✅ Done | LogReg AUC 0.772 on test |
| 5. Model comparison | ✅ Done | RF AUC 0.797 on test |
| 6. EDA notebook | ✅ Done | 6 sections, 6 figures |
| 7. Baseline notebook | ✅ Done | Full pipeline + threshold analysis |
| 8. RF notebook | ✅ Done | Comparison + feature importances |
| 9. Hyperparameter tuning | 🔄 Next | RandomizedSearchCV on RF |
| 10. Tests | ⬜ Planned | pytest for `features.py` |
| 11. MLflow tracking | ⬜ Planned | Experiment logging |
| 12. README final | ⬜ Planned | Results, usage, figures |

---

## Session Log

### Session 1
- Created repo on GitHub
- Added README with problem statement and methodology outline
- Created `src/` module structure

### Session 2
- Wrote `data.py`: `load_data()` and `split_data()`
- Fixed `.xls` loading issue (`engine="xlrd"`, `skiprows=1`)
- Recoded undocumented EDUCATION / MARRIAGE categories
- Validated split sizes and stratification

### Session 3
- Wrote `features.py`: `build_features()` and `scale_features()`
- Engineered 19 features across 4 groups
- Confirmed no data leakage in scaling logic

### Session 4
- Wrote `models.py`: `train_logistic_regression()`, `train_random_forest()`
- Wrote `evaluation.py`: `evaluate_model()` with ROC-AUC, Precision, Recall, F1
- Wrote `train.py`: end-to-end pipeline orchestrator

### Session 5
- Completed `eda.ipynb`: target distribution, demographics, distributions,
  repayment status, engineered features, correlation matrix
- Generated figures 01–06 in `reports/figures/`

### Session 6
- Completed `baseline_logreg.ipynb`: full pipeline, metrics table,
  ROC/PR curves, confusion matrix, threshold sweep, coefficients
- Generated figures 07–10

### Session 7
- Completed `02_random_forest.ipynb`: side-by-side comparison,
  overlaid curves, feature importances, overfitting diagnosis
- Generated figures 11–13

---

## Current Metrics (Test Set)

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7723 | 0.8817 | 0.7849 | 0.8305 |
| Random Forest | 0.7973 | 0.8824 | 0.8391 | 0.8602 |

---

## Known Issues / Tech Debt

- RF has a train-test AUC gap of ~0.10 — needs hyperparameter tuning
- No unit tests yet for `build_features()`
- No experiment tracking — runs are not logged systematically
- `reports/figures/` not committed to git (should add to `.gitignore` or use Git LFS)