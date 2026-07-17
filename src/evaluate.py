"""
Loads a saved pipeline, re-evaluates on a held-out split, and writes
ROC curve, precision-recall curve, and confusion matrix plots to
reports/figures/. Also writes a SHAP summary plot if `shap` is installed
and the final estimator is tree-based.

Run:
    python src/evaluate.py --data data/insurance_claims.csv --model models/fraud_pipeline.pkl
"""
import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot — no GUI backend in CI
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src.features import LABEL  # noqa: E402

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.data)
    X = df.drop(columns=[LABEL])
    y = df[LABEL]
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    pipe = joblib.load(args.model)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve — Fraud Detection")
    fig.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve — Fraud Detection")
    fig.savefig(os.path.join(args.out_dir, "precision_recall_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ROC, PR, and confusion matrix plots to {args.out_dir}/")

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        try:
            feature_names = pipe.named_steps["prep"].get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(len(importances))]
        top = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)[:15]
        names, vals = zip(*top)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.barplot(x=list(vals), y=list(names), ax=ax, color="#4C72B0")
        ax.set_title("Top 15 Feature Importances")
        fig.savefig(os.path.join(args.out_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved feature_importance.png")

    if HAS_SHAP and hasattr(clf, "predict_proba"):
        try:
            X_test_transformed = pipe.named_steps["prep"].transform(X_test)
            explainer = shap.Explainer(clf, X_test_transformed)
            shap_values = explainer(X_test_transformed[:200])
            shap.summary_plot(shap_values, show=False)
            plt.savefig(os.path.join(args.out_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved shap_summary.png")
        except Exception as e:
            print(f"[evaluate] SHAP plot skipped: {e}")
    elif not HAS_SHAP:
        print("[evaluate] shap not installed — skipping SHAP plot (pip install shap to enable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/insurance_claims.csv")
    parser.add_argument("--model", type=str, default="models/fraud_pipeline.pkl")
    parser.add_argument("--out-dir", type=str, default="reports/figures")
    args = parser.parse_args()
    main(args)