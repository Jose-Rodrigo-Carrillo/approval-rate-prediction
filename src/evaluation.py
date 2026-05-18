# src/evaluation.py

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """
    Evaluate a trained classifier using business-relevant metrics.

    Why these metrics
    -----------------
    - ROC-AUC   : threshold-independent ranking quality; handles class imbalance
    - Precision : of all predicted "approved", how many truly paid? (risk control)
    - Recall    : of all true payers, how many did we catch? (conversion)
    - F1        : harmonic mean — useful summary when imbalance is present

    Parameters
    ----------
    model     : fitted sklearn model with predict_proba()
    X         : feature matrix (scaled np.ndarray or DataFrame)
    y         : true labels
    threshold : decision threshold for positive class (default 0.5)

    Returns
    -------
    dict with keys: roc_auc, precision, recall, f1
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "roc_auc":   roc_auc_score(y, probs),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "f1":        f1_score(y, preds, zero_division=0),
    }