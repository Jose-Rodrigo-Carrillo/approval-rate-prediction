# src/train.py
"""
End-to-end training pipeline — Approval Rate Prediction.

Trains two models (Logistic Regression baseline + Random Forest),
evaluates both on train / val / test, and prints a side-by-side comparison.

Usage
-----
    cd src
    python train.py

    # Custom path:
    python train.py --data "../data/raw/default of credit card clients.xls"
"""

import argparse
from pathlib import Path

from data       import load_data, split_data
from features   import build_features, scale_features
from models     import train_logistic_regression, train_random_forest
from evaluation import evaluate_model

DATA_PATH_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "default of credit card clients.xls"
)


def run_pipeline(data_path: str):
    print("\n" + "=" * 65)
    print("  APPROVAL RATE PREDICTION — Training Pipeline")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # 1. Load & clean                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n[1/6] Loading data ...")
    df = load_data(data_path)
    approval_rate = df["APPROVED"].mean()
    print(f"      Rows: {len(df):,}  |  Raw features: {df.shape[1] - 1}")
    print(f"      Approval rate: {approval_rate:.2%}  "
          f"(imbalance ≈ {approval_rate / (1 - approval_rate):.1f}:1)")

    # ------------------------------------------------------------------ #
    # 2. Feature engineering                                               #
    # ------------------------------------------------------------------ #
    print("\n[2/6] Engineering features ...")
    df_fe = build_features(df)
    n_new = df_fe.shape[1] - df.shape[1]
    print(f"      Features after engineering: {df_fe.shape[1] - 1}  (+{n_new} new)")

    # ------------------------------------------------------------------ #
    # 3. Split                                                             #
    # ------------------------------------------------------------------ #
    print("\n[3/6] Splitting train / val / test ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_fe)
    print(f"      Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------ #
    # 4. Scale (fit on train only — no leakage)                            #
    # ------------------------------------------------------------------ #
    print("\n[4/6] Scaling features ...")
    X_tr_s, X_val_s, X_te_s, scaler = scale_features(X_train, X_val, X_test)
    print(f"      Feature matrix: {X_tr_s.shape}")

    # ------------------------------------------------------------------ #
    # 5. Train both models                                                 #
    # ------------------------------------------------------------------ #
    print("\n[5/6] Training models ...")

    print("      → Logistic Regression ...", end=" ", flush=True)
    logreg = train_logistic_regression(X_tr_s, y_train)
    print("done")

    print("      → Random Forest ...", end=" ", flush=True)
    rf = train_random_forest(X_tr_s, y_train)
    print("done")

    # ------------------------------------------------------------------ #
    # 6. Compare on all splits                                             #
    # ------------------------------------------------------------------ #
    print("\n[6/6] Evaluation\n")

    models = {"LogReg (baseline)": logreg, "Random Forest": rf}
    splits = {
        "Train": (X_tr_s,  y_train),
        "Val":   (X_val_s, y_val),
        "Test":  (X_te_s,  y_test),
    }

    col_w = 20
    header = f"  {'Model':<{col_w}} {'Split':<8} {'AUC':>7} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_results = {}
    for model_name, model in models.items():
        for split_name, (X, y) in splits.items():
            m = evaluate_model(model, X, y)
            all_results[(model_name, split_name)] = m
            print(
                f"  {model_name:<{col_w}} {split_name:<8} "
                f"{m['roc_auc']:>7.4f} "
                f"{m['precision']:>10.4f} "
                f"{m['recall']:>8.4f} "
                f"{m['f1']:>8.4f}"
            )
        print()

    # Summary: which model wins on test AUC?
    lr_auc  = all_results[("LogReg (baseline)", "Test")]["roc_auc"]
    rf_auc  = all_results[("Random Forest",     "Test")]["roc_auc"]
    delta   = rf_auc - lr_auc
    winner  = "Random Forest" if delta > 0 else "Logistic Regression"

    print(f"  Best model on Test AUC: {winner}  "
          f"(ΔAUCtest = {delta:+.4f})")

    # Overfitting check for each model
    print()
    for model_name in models:
        gap = (all_results[(model_name, "Train")]["roc_auc"]
               - all_results[(model_name, "Test")]["roc_auc"])
        flag = "⚠️  possible overfit" if gap > 0.05 else "✅ generalises well"
        print(f"  {model_name:<{col_w}}  train-test AUC gap = {gap:+.4f}  {flag}")

    print("\n" + "=" * 65 + "\n")
    return {"logreg": logreg, "rf": rf}, scaler, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_PATH_DEFAULT),
        help="Path to the raw .xls dataset",
    )
    args = parser.parse_args()
    run_pipeline(args.data)