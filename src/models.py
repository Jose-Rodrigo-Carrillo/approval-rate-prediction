# src/models.py
"""
Model definitions for the Approval Rate Prediction project.

Design principles
-----------------
- Each function trains ONE model and returns it fitted.
- No data splitting or scaling happens here — that belongs in train.py.
- class_weight="balanced" is used throughout: the dataset has a 78/22
  class imbalance, so ignoring it would bias the model toward always
  predicting "approved".
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model.

    Why this is the baseline
    ------------------------
    - Fast to train and fully interpretable via coefficients.
    - Linear decision boundary: if a tree-based model beats it significantly,
      that tells us the relationship between features and target is non-linear.
    - class_weight="balanced" adjusts for the 78/22 imbalance automatically.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Scaled training features (output of scale_features).
    y_train : array-like, shape (n_samples,)
        Binary target (1 = approved, 0 = defaulted).

    Returns
    -------
    LogisticRegression  — fitted model
    """
    model = LogisticRegression(
        max_iter=1000,          # default 100 often fails to converge on this dataset
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Why Random Forest as the second model
    --------------------------------------
    - Captures non-linear interactions (e.g. high utilisation + delayed payment).
    - Robust to feature scale: does NOT require StandardScaler, but passing
      scaled features does not hurt it either.
    - feature_importances_ provides a direct ranking of which engineered
      features matter most — useful for the EDA notebook.
    - n_estimators=300: enough trees for stable estimates without excessive
      training time on 21k rows.
    - min_samples_leaf=20: prevents individual trees from memorising noise;
      acts as implicit regularisation.

    Parameters
    ----------
    X_train : array-like
        Training features (scaled or unscaled — RF is scale-invariant).
    y_train : array-like
        Binary target.

    Returns
    -------
    RandomForestClassifier  — fitted model
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,         # let trees grow fully; min_samples_leaf controls depth
        min_samples_leaf=20,    # each leaf must contain ≥20 samples (regularisation)
        class_weight="balanced",
        n_jobs=-1,              # use all available CPU cores
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model