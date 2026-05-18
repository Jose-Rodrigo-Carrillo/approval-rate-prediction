# src/data.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# UCI dataset has a double-header: row 0 = X1..Y aliases, row 1 = real column names
_SKIPROWS = 1

# Values not documented in the original UCI codebook — merged into bucket 3 ("Other")
_EDUCATION_UNKNOWN = {0, 4, 5, 6}
_MARRIAGE_UNKNOWN  = {0}


def load_data(path: str) -> pd.DataFrame:
    """
    Load the UCI Credit Card Default dataset (.xls) and return a clean DataFrame.

   
    """
    df = pd.read_excel(Path(path), engine="xlrd", skiprows=_SKIPROWS)

    # Drop row identifier
    df = df.drop(columns=["ID"])

    # Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
    )

    # Recode undocumented categories → 3 ("Other") per UCI docs convention
    df["EDUCATION"] = df["EDUCATION"].apply(
        lambda x: 3 if x in _EDUCATION_UNKNOWN else x
    )
    df["MARRIAGE"] = df["MARRIAGE"].apply(
        lambda x: 3 if x in _MARRIAGE_UNKNOWN else x
    )

    # Binary target: 1 = paid on time (approved), 0 = defaulted
    df["APPROVED"] = (1 - df["DEFAULT_PAYMENT_NEXT_MONTH"]).astype(int)
    df = df.drop(columns=["DEFAULT_PAYMENT_NEXT_MONTH"])

    return df


def split_data(
    df: pd.DataFrame,
    target: str = "APPROVED",
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """
    Stratified three-way split: train (70%) / val (15%) / test (15%).

   

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test