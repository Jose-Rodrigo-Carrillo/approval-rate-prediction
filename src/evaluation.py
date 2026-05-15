# src/evaluation.py

from sklearn.metrics import roc_auc_score, precision_score, recall_score


def evaluate_model(model, X, y, threshold: float = 0.5):
    """
    Evaluate a trained classification model using business-relevant metrics.

    Parameters
    ----------
    model : fitted model
        Trained classification model with predict_proba method.
    X : array-like
        Feature matrix.
    y : array-like
        True labels.
    threshold : float, default=0.5
        Decision threshold for converting probabilities into class predictions.

    Returns
    -------
    metrics : dict
        Dictionary with ROC-AUC, precision and recall.
    """
    # Predicted probabilities for the positive class
    probs = model.predict_proba(X)[:, 1]

    # Class predictions based on threshold
    preds = (probs >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, probs),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
    }

    return metrics