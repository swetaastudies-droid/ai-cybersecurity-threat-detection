"""Train the supervised classifier and optional anomaly detector."""
from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_loader import load_train_test
from src.data_cleaning import clean_dataset
from src.feature_engineering import split_features_target
from src.anomaly_detection import train_anomaly_detector

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def main():
    MODELS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    train_df, test_df = load_train_test()
    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    X_train, X_test, y_train, y_test, preprocessor, feature_columns = (
        split_features_target(train_df, test_df)
    )

    classifier = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training Random Forest classifier...")
    classifier.fit(X_train, y_train)
    joblib.dump({
        "model": classifier,
        "feature_columns": feature_columns,
    }, MODELS / "threat_detection_model.joblib")

    # Fit a separate anomaly model on benign training rows only.
    # This model uses the raw feature DataFrame and its own fitted preprocessor.
    print("Training optional Isolation Forest...")
    train_anomaly_detector(X_train, y_train, preprocessor)

    # Save test features and labels for evaluation; test data is not fitted.
    X_test.to_csv(OUTPUTS / "test_features.csv", index=False)
    y_test.to_csv(OUTPUTS / "test_labels.csv", index=False)

    with open(OUTPUTS / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "classifier": "RandomForestClassifier",
            "anomaly_detector": "IsolationForest",
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
            "feature_count": len(feature_columns),
            "random_state": 42,
        }, f, indent=2)

    print("Training complete.")
    print(f"Saved classifier: {MODELS / 'threat_detection_model.joblib'}")
    print(f"Saved anomaly model: {MODELS / 'anomaly_detection_model.joblib'}")


if __name__ == "__main__":
    main()
