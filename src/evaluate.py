"""Evaluate a saved model: classification report + ROC/PR curve PNGs."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)

from src.data import load_dataframe, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model.joblib"))
    parser.add_argument("--data", type=Path, default=Path("data/ai4i2020.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    df = load_dataframe(args.data)
    _, X_test, _, y_test = split(df)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"Threshold: {threshold:.2f}")
    print(classification_report(y_test, y_pred, target_names=["no_failure", "failure"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc)
    ax_roc.set_title("ROC Curve — Predictive Maintenance")
    fig_roc.tight_layout()
    fig_roc.savefig(args.out_dir / "roc_curve.png", dpi=120)

    fig_pr, ax_pr = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax_pr)
    ax_pr.set_title("Precision-Recall Curve — Predictive Maintenance")
    fig_pr.tight_layout()
    fig_pr.savefig(args.out_dir / "pr_curve.png", dpi=120)

    print(f"\nCurves saved to {args.out_dir}/roc_curve.png and pr_curve.png")


if __name__ == "__main__":
    main()
