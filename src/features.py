# src/features.py
"""
Feature engineering for the UCI Credit Card Default dataset.

Column reference
----------------
LIMIT_BAL               : Credit limit (NT dollars)
SEX                     : 1=male, 2=female
EDUCATION               : 1=grad, 2=university, 3=high school / other
MARRIAGE                : 1=married, 2=single, 3=other
AGE                     : Age in years
PAY_0..PAY_6            : Repayment status months 1-6
                          -2=no consumption, -1=paid in full, 0=revolving credit
                          1..8=months of payment delay
BILL_AMT1..BILL_AMT6    : Bill statement amount months 1-6
PAY_AMT1..PAY_AMT6      : Previous payment amount months 1-6
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Repayment-status columns ordered most-recent → oldest
PAY_COLS      = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS     = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS  = ["PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",  "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6"]

# Features that will be log-transformed (right-skewed, non-negative after shift)
_LOG_FEATURES = [
    "LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the raw UCI columns.

    All operations are pure transforms (no fit required) so this function
    can be called on train, val, and test sets identically — no leakage risk.

    New columns created
    -------------------
    UTILIZATION_1..3        : bill / limit ratio (months 1-3)
    PAY_RATIO_1..3          : payment / bill ratio (months 1-3) — how much was repaid
    MAX_DELAY               : worst repayment status across all 6 months
    N_DELAYED               : number of months with payment delay (PAY > 0)
    N_DULY_PAID             : number of months paid in full (PAY == -1)
    BILL_TREND              : slope of bill amounts over 6 months (positive = growing debt)
    AVG_BILL                : mean bill across 6 months
    AVG_PAY_AMT             : mean payment amount across 6 months
    LOG_LIMIT_BAL           : log1p of credit limit
    LOG_BILL_AMT1..3        : log1p of bill amounts (handles negatives via clip)
    LOG_PAY_AMT1..3         : log1p of payment amounts

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (output of load_data, before splitting).

    Returns
    -------
    pd.DataFrame with original columns + engineered features.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Credit utilisation: how much of the limit is being used
    #    Clipped to [0, 5] to handle negative bills (credits/refunds)
    # ------------------------------------------------------------------
    for i, bill_col in enumerate(BILL_COLS[:3], start=1):
        df[f"UTILIZATION_{i}"] = (
            df[bill_col] / df["LIMIT_BAL"]
        ).clip(0, 5)

    # ------------------------------------------------------------------
    # 2. Repayment ratio: what fraction of the bill was actually paid
    #    Avoid division by zero; clip to [0, 2] (>1 = overpayment)
    # ------------------------------------------------------------------
    for i, (bill_col, pay_col) in enumerate(
        zip(BILL_COLS[:3], PAY_AMT_COLS[:3]), start=1
    ):
        denom = df[bill_col].clip(lower=1)          # 1 NT$ floor avoids /0
        df[f"PAY_RATIO_{i}"] = (df[pay_col] / denom).clip(0, 2)

    # ------------------------------------------------------------------
    # 3. Delay behaviour across 6 months
    # ------------------------------------------------------------------
    pay_matrix = df[PAY_COLS]

    df["MAX_DELAY"]    = pay_matrix.max(axis=1)          # worst single month
    df["N_DELAYED"]    = (pay_matrix > 0).sum(axis=1)    # months with delay
    df["N_DULY_PAID"]  = (pay_matrix == -1).sum(axis=1)  # months fully paid

    # ------------------------------------------------------------------
    # 4. Bill trend: positive = debt is growing, negative = shrinking
    #    Simple OLS slope across 6 time points (most recent = index 0)
    # ------------------------------------------------------------------
    bill_matrix = df[BILL_COLS].values.astype(float)     # shape (n, 6)
    x = np.arange(6, dtype=float)
    x -= x.mean()
    # slope = cov(x, y) / var(x) — vectorised across all rows
    df["BILL_TREND"] = (bill_matrix @ x) / (x @ x)

    # ------------------------------------------------------------------
    # 5. Simple aggregates
    # ------------------------------------------------------------------
    df["AVG_BILL"]    = df[BILL_COLS].mean(axis=1)
    df["AVG_PAY_AMT"] = df[PAY_AMT_COLS].mean(axis=1)

    # ------------------------------------------------------------------
    # 6. Log transforms for skewed continuous features
    #    BILL_AMT can be negative (refunds) → shift to 0 before log1p
    # ------------------------------------------------------------------
    df["LOG_LIMIT_BAL"] = np.log1p(df["LIMIT_BAL"])

    for i, col in enumerate(["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"], start=1):
        df[f"LOG_BILL_AMT{i}"] = np.log1p(df[col].clip(lower=0))

    for i, col in enumerate(["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"], start=1):
        df[f"LOG_PAY_AMT{i}"] = np.log1p(df[col].clip(lower=0))

    return df


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
):
    """
    Fit StandardScaler on training set only; transform all three splits.

    Fitting on val/test would be data leakage — the scaler must only
    see the training distribution.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
    scaler                                       : fitted StandardScaler
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler