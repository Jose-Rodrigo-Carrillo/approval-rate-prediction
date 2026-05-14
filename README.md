# Approval-rate-prediction

Machine learning model to predict transaction approval probability using business features.

# Approval Rate Prediction

## Overview
This project builds a machine learning model to predict whether a transaction will be approved
or declined based on transaction and contextual features.

The objective is to improve approval rate monitoring while balancing risk and conversion.

## Business Problem
- Can we predict transaction approval outcomes before authorization?
- Which factors have the strongest impact on approval probability?
- How can this insight improve conversion without increasing risk?

## Dataset

This project uses the **UCI Credit Card Default dataset** as a proxy for
transaction approval outcomes.

- **Source:** UCI Machine Learning Repository
- **Original target:** `default_payment_next_month`
- **Adapted target:**
  - `approved = 1 - default_payment_next_month`

### Rationale
In a real payment authorization context, transactions that are predicted
to default would typically be declined or flagged as high risk.
This dataset provides a reasonable public proxy to study:
- approval probability
- class imbalance
- risk vs conversion trade-offs

The dataset is publicly available and contains no sensitive or personal
identifiable information.

## Project Structure


approval-rate-prediction/
├── data/          # Raw and processed datasets
├── notebooks/     # EDA and exploratory analysis
├── src/           # Reproducible training and evaluation code
├── docs/          # Technical decisions and experiment tracking
└── tests/         # Minimal tests for data integrity


## Methodology

1. Data loading and validation
2. Train / validation / test split with stratification
3. Baseline model using Logistic Regression
4. Evaluation using ROC-AUC and Precision/Recall
5. Threshold optimization based on business constraints
6. Model interpretability and feature analysis


## Why These Metrics
Accuracy alone is not sufficient due to class imbalance.
ROC-AUC and Precision/Recall provide a clearer view of model performance in a payments context.

## Results
- The baseline model provides a robust and interpretable starting point
- Key drivers include transaction amount, country, and payment method
- The model highlights clear trade-offs between approval rate and risk control

## Tools Used
- Python (pandas, scikit-learn)
- Matplotlib / Seaborn

## Next Steps
- Threshold optimization based on business risk tolerance
- Model monitoring over time
- Integration into real-time decision pipelines
