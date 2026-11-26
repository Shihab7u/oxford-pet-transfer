"""Train scikit-learn classifiers on frozen MobileNetV3-Small features.

Scaling and hyper-parameter selection use the training and validation splits only;
the official test split is evaluated exactly once for the selected configuration.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pet_data import class_names
from utils import FEATURES_DIR, METRICS_DIR, PRED_DIR, RESULTS_DIR, TABLES_DIR, compute_metrics, save_json, seed_everything

GRID = [1e-3, 1e-2, 1e-1, 1.0]


def candidates(seed: int):
    return {
        "logreg": [
            (
                f"C={c}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, max_iter=2000, random_state=seed, n_jobs=-1),
                ),
            )
            for c in GRID
        ],
        "linear_svm": [
            (
                f"C={c}",
                make_pipeline(StandardScaler(), LinearSVC(C=c, max_iter=10000, random_state=seed)),
            )
            for c in GRID
        ],
        "random_forest": [
            (
                "500 trees, sqrt features",
                make_pipeline(
                    StandardScaler(),
                    RandomForestClassifier(
                        n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=seed
                    ),
                ),
            )
        ],
    }


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba.argmax(1), proba.max(1)
    scores = model.decision_function(X)
    return scores.argmax(1), scores.max(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    seed_everything(args.seed)
    names = class_names(args.data_root)

    data = np.load(FEATURES_DIR / "mobilenetv3small_features.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    files_test = data["files_test"]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for family, options in candidates(args.seed).items():
        trials = []
        for label, model in options:
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            val_pred = model.predict(X_val)
            val_metrics = compute_metrics(y_val, val_pred, names)
            trials.append(
                {
                    "config": label,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "fit_seconds": fit_seconds,
                }
            )
            print(f"{family:14s} {label:28s} val acc {val_metrics['accuracy']:.4f} "
                  f"macroF1 {val_metrics['macro_f1']:.4f} ({fit_seconds:.1f}s)", flush=True)

        best = max(trials, key=lambda t: t["val_macro_f1"])
        best_model = dict(options)[best["config"]]

        start = time.perf_counter()
        y_pred, confidence = predict_scores(best_model, X_test)
        predict_seconds = time.perf_counter() - start
        test_metrics = compute_metrics(y_test, y_pred, names)
        train_pred = best_model.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred, names)

        result = {
            "family": family,
            "selected_config": best["config"],
            "selection_rule": "best validation macro-F1 over the grid",
            "trials": trials,
            "fit_seconds": best["fit_seconds"],
            "predict_seconds": predict_seconds,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": best["val_accuracy"],
            "val_macro_f1": best["val_macro_f1"],
            "test": test_metrics,
            "n_features": int(X_train.shape[1]),
            "scaler": "StandardScaler fit on training features only",
        }
        save_json(result, METRICS_DIR / f"sklearn_{family}.json")

        pd.DataFrame(
            {
                "filename": files_test,
                "label": y_test,
                "prediction": y_pred,
                "confidence": confidence,
                "true_name": [names[i] for i in y_test],
                "pred_name": [names[i] for i in y_pred],
            }
        ).to_csv(PRED_DIR / f"sklearn_{family}_test_predictions.csv", index=False)

        summary.append(
            {
                "approach": family,
                "config": best["config"],
                "trainable_params": int(X_train.shape[1]) * len(names) + len(names),
                "train_seconds": best["fit_seconds"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": best["val_accuracy"],
                "val_macro_f1": best["val_macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
        print(f"  -> test acc {test_metrics['accuracy']:.4f} macroF1 {test_metrics['macro_f1']:.4f}", flush=True)

    pd.DataFrame(summary).to_csv(TABLES_DIR / "sklearn_summary.csv", index=False)


if __name__ == "__main__":
    main()
