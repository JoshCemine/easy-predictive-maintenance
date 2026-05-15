"""Train a predictive-maintenance classifier and write artifacts to disk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data import load_dataframe, split
from src.features import build_preprocessor

PARAM_GRID = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [2, 3],
    "classifier__learning_rate": [0.05, 0.1],
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    )


def tune_threshold(y_true, y_proba) -> tuple[float, dict]:
    """Pick threshold maximising F1 on the positive (failure) class."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1, best_thr, best_cm = -1.0, 0.5, None
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return best_thr, {"f1": best_f1, "confusion_matrix": best_cm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ai4i2020.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("model.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("metrics.json"))
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()

    df = load_dataframe(args.data)
    X_train, X_test, y_train, y_test = split(df)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Positive rate (train): {y_train.mean():.3%}")

    pipeline = build_pipeline()
    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring="average_precision",
        cv=args.cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"Best params: {search.best_params_}")

    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    threshold, threshold_metrics = tune_threshold(y_test, y_proba)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "best_params": search.best_params_,
        "threshold": threshold,
        "at_threshold": threshold_metrics,
        "positive_rate_test": float(y_test.mean()),
    }

    joblib.dump({"pipeline": best_pipeline, "threshold": threshold}, args.model_out)
    args.metrics_out.write_text(json.dumps(metrics, indent=2))

    print("\n=== Results ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {args.model_out}")
    print(f"Metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main()
