# src/features.py

from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_val, X_test):
    """
    Scale numerical features using StandardScaler.

    The scaler is fitted ONLY on the training set to avoid data leakage.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler