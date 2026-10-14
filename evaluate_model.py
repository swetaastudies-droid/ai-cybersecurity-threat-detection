"""Evaluate the saved classifier on the held-out test set."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    artifact = joblib.load(ROOT / "models" / "threat_detection_model.joblib")
    model = artifact["model"]
    X_test = pd.read_csv(OUTPUTS / "test_features.csv")
    y_test = pd.read_csv(OUTPUTS / "test_labels.csv").iloc[:, 0].astype(int)

    y_pred = model.predict(X_test).astype(int)
    classes = list(model.named_steps["classifier"].classes_)
    attack_idx = classes.index(1)
    attack_scores = model.predict_proba(X_test)[:, attack_idx]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, labels=[0, 1],
            target_names=["Benign", "Attack"], output_dict=True,
            zero_division=0,
        ),
    }
    if y_test.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, attack_scores))

    OUTPUTS.mkdir(exist_ok=True)
    with open(OUTPUTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "actual_label": y_test,
        "predicted_label": y_pred,
        "attack_probability": attack_scores,
    }).to_csv(OUTPUTS / "predictions.csv", index=False)

    print("Evaluation metrics (measured on held-out test data):")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")
    print("\nConfusion matrix (rows=actual [0,1], columns=predicted [0,1]):")
    print(np.array(metrics["confusion_matrix"]))
    print("\n", classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Benign", "Attack"], zero_division=0
    ))


if __name__ == "__main__":
    main()
