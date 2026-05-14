# Technical Decisions

## Dataset Selection

**Chosen dataset:** UCI Credit Card Default Dataset

### Alternatives considered
- Santander Transaction Dataset (Kaggle)
- Synthetic payment datasets

### Decision rationale
The UCI dataset was selected because:
- It represents a realistic risk-related outcome (default)
- It is publicly available and widely referenced
- It allows meaningful discussion of imbalance, risk and approval trade-offs

The target variable was adapted as:
approved = 1 - default_payment_next_month

This transformation reflects how high-risk transactions are typically
declined in real-world payment systems.

---

## Validation Strategy

- Data is split into train / validation / test sets
- Splitting is performed **before any preprocessing**
- Stratification is used to preserve class imbalance

This prevents data leakage and ensures unbiased evaluation.

---

## Model Choice

**Baseline model:** Logistic Regression

### Reasons
- High interpretability
- Stable baseline for imbalanced classification
- Commonly used in risk and credit scoring contexts

More complex models are considered only after establishing a strong baseline.