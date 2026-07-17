"""
Trains and compares candidate models on the fused multimodal features,
selects the best by cross-validated ROC-AUC, refits on the full training
set, evaluates on a held-out test set, and saves the fitted pipeline +
a metrics report.

Run:
    python src/train.py --data data/insurance_claims.csv
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.features import LABEL, build_preprocessor

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_candidate_models():
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0, solver="liblinear"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="auc",
            random_state=42,
        )
    else:
        print("[train] xgboost not installed — skipping (pip install xgboost to enable)")
    return models


def main(args):
    df = pd.read_csv(args.data)
    X = df.drop(columns=[LABEL])
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Fraud rate (train): {y_train.mean():.3%}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted = {}

    for name, model in get_candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        results[name] = {"cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std())}
        print(f"{name:20s} CV ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best_pipe = fitted[best_name]
    print(f"\nBest model by CV ROC-AUC: {best_name}")

    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = best_pipe.predict(X_test)
    test_auc = roc_auc_score(y_test, y_proba)
    test_ap = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nTest ROC-AUC: {test_auc:.4f}   Test PR-AUC (avg precision): {test_ap:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", np.array(cm))

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "fraud_pipeline.pkl")
    joblib.dump(best_pipe, model_path)

    metrics = {
        "cv_results": results,
        "selected_model": best_name,
        "test_roc_auc": test_auc,
        "test_pr_auc": test_ap,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/insurance_claims.csv")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()
    main(args)
