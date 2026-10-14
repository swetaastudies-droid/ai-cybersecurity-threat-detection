"""Load a trained model and predict attack labels for new CSV records."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "threat_detection_model.joblib"


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run: python train_model.py")
    return joblib.load(MODEL_PATH)


def predict_records(records: pd.DataFrame) -> pd.DataFrame:
    """Return predicted label and model's attack-class score for each row."""
    if records.empty:
        raise ValueError("Input contains no records.")

    artifact = load_model_artifact()
    model = artifact["model"]
    columns = artifact["feature_columns"]
    missing = sorted(set(columns) - set(records.columns))
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    X = records[columns].copy()
    labels = model.predict(X).astype(int)
    probabilities = model.predict_proba(X)
    classes = list(model.named_steps["classifier"].classes_)
    attack_index = classes.index(1)

    return pd.DataFrame({
        "predicted_label": labels,
        "attack_probability": probabilities[:, attack_index],
    }, index=records.index)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict attacks in a CSV file.")
    parser.add_argument("csv_path", help="CSV containing the model's feature columns")
    parser.add_argument("--output", default="outputs/new_predictions.csv")
    args = parser.parse_args()

    records = pd.read_csv(args.csv_path)
    results = predict_records(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.head())
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
