# src/data.py

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw data and create target variable.
    """
    df = pd.read_csv(path)
    df["approved"] = 1 - df["default_payment_next_month"]
    return df


def split_data(
    df: pd.DataFrame,
    target: str = "approved",
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """
    Split data into train, validation and test sets.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second split: train vs validation
    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test