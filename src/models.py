# src/models.py

from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model as baseline.

    Parameters
    ----------
    X_train : array-like
        Training features (scaled).
    y_train : array-like
        Training target.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model