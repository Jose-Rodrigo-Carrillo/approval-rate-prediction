# Project Roadmap

Structured plan from current state to a production-ready,
portfolio-quality Data Science project.

Each phase maps to the target skill stack for a DS role in payments.

---

## Phase 1 — Foundation ✅ Complete

**Goal:** Working end-to-end pipeline with a documented baseline.

- [x] Repo structure (`src/`, `notebooks/`, `data/`, `reports/`)
- [x] `data.py` — load, clean, split with no leakage
- [x] `features.py` — 19 engineered features with domain rationale
- [x] `models.py` — LogReg baseline + Random Forest
- [x] `evaluation.py` — ROC-AUC, Precision, Recall, F1
- [x] `train.py` — orchestrated pipeline
- [x] `eda.ipynb` — full exploratory analysis
- [x] `baseline_logreg.ipynb` — baseline model documented
- [x] `02_random_forest.ipynb` — model comparison

**Result:** LogReg AUC 0.772 → RF AUC 0.797 on test set.

---

## Phase 2 — Model Quality 🔄 In Progress

**Goal:** Reduce RF overfitting and establish a tuned, well-calibrated model.

**Skills covered:** cross-validation, hyperparameter tuning, calibration

- [ ] `03_hyperparameter_tuning.ipynb`
  - `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)`
  - Search space: `max_depth`, `min_samples_leaf`, `n_estimators`, `max_features`
  - Target: reduce train-test AUC gap from ~0.10 to <0.05
- [ ] Probability calibration (`CalibratedClassifierCV`)
  - Check calibration curve — are predicted probabilities reliable?
- [ ] Update `models.py` with tuned RF function
- [ ] Update `train.py` to log best hyperparameters

**Expected outcome:** Tuned RF AUC ~0.800–0.810, gap <0.05.

---

## Phase 3 — Explainability

**Goal:** Make the model interpretable for a business audience.

**Skills covered:** SHAP, feature importance, individual prediction explanation

- [ ] `04_explainability.ipynb`
  - SHAP values for the tuned RF
  - Summary plot: global feature importance
  - Waterfall plot: individual prediction explanation
  - "Why was this client declined?" — business-readable output
- [ ] Add `shap` to `requirements.txt`

---

## Phase 4 — Robustness & Testing

**Goal:** Make the codebase trustworthy and reproducible.

**Skills covered:** pytest, data validation, reproducibility

- [ ] `tests/test_features.py`
  - Test `build_features()` output shape
  - Test no nulls introduced by engineering
  - Test UTILIZATION clipping bounds
  - Test LOG features are non-negative
- [ ] `tests/test_data.py`
  - Test split sizes and stratification ratios
  - Test target column creation
- [ ] `Great Expectations` or manual data validation in `data.py`
  - Assert expected column names on load
  - Assert value ranges for PAY columns (-2 to 8)
  - Assert no nulls after cleaning

---

## Phase 5 — Experiment Tracking (MLOps intro)

**Goal:** Log and compare experiments systematically.

**Skills covered:** MLflow, experiment management

- [ ] Add `mlflow` to `requirements.txt`
- [ ] Instrument `train.py` with MLflow run logging:
  - Parameters: model type, hyperparameters, threshold
  - Metrics: AUC, F1, Precision, Recall per split
  - Artifacts: trained model, scaler, feature list
- [ ] `05_mlflow_experiments.ipynb`
  - Compare runs in MLflow UI
  - Select best run by validation AUC

---

## Phase 6 — Portfolio Finalisation

**Goal:** Make the repo readable and impressive to a hiring manager or DS team.

- [ ] `README.md` final version
  - Project summary and business problem
  - Architecture diagram (`src/` module relationships)
  - Results table with figures embedded
  - How to reproduce (setup, run instructions)
  - What I would do next with more time/data
- [ ] `requirements.txt` pinned versions
- [ ] `.gitignore` — exclude `data/raw/`, `reports/figures/`, `.ipynb_checkpoints/`
- [ ] Clean commit history — one commit per meaningful phase

---

## Skill Coverage Map

| Skill (target DS stack) | Covered by |
|---|---|
| Python / pandas | All modules |
| Feature engineering | `features.py`, Phase 1 |
| sklearn pipelines | `train.py`, Phase 2 |
| Cross-validation | Phase 2 |
| Experiment tracking | Phase 5 (MLflow) |
| Model explainability | Phase 3 (SHAP) |
| Testing | Phase 4 (pytest) |
| Data validation | Phase 4 (Great Expectations) |
| Reproducible pipelines | Phase 6 |
| Anomaly / time-series | Future project |
| SQL advanced | Separate track |