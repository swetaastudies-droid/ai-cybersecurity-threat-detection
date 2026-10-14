"""Run an offline demonstration using held-out benchmark records and synthetic metadata."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.detection import predict_records
from src.anomaly_detection import score_anomalies
from src.alert_engine import create_alerts
from src.simulation import generate_simulated_events

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    OUTPUTS.mkdir(exist_ok=True)
    features_path = OUTPUTS / "test_features.csv"
    labels_path = OUTPUTS / "test_labels.csv"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Test features/labels missing. Run python train_model.py first."
        )

    # Use a small sample for a quick, reproducible demonstration.
    features = pd.read_csv(features_path).head(2000)
    labels = pd.read_csv(labels_path).iloc[:len(features), 0].astype(int)

    classifier_predictions = predict_records(features)
    alerts = create_alerts(classifier_predictions, threshold=0.50)
    anomaly_results = score_anomalies(features)

    benchmark_results = pd.concat(
        [
            pd.DataFrame({
                "actual_label": labels.to_numpy(),
            }),
            alerts.reset_index(drop=True),
            anomaly_results.reset_index(drop=True),
        ],
        axis=1,
    )
    benchmark_results.to_csv(OUTPUTS / "simulation_benchmark_results.csv", index=False)

    # These are fictional event descriptions, not model-ready flow records.
    synthetic_events = generate_simulated_events(20)
    synthetic_events.to_csv(OUTPUTS / "simulated_event_metadata.csv", index=False)

    # Graph 1: distribution of Isolation Forest decision scores.
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=benchmark_results,
        x="anomaly_score",
        hue="is_anomaly",
        bins=35,
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.title("Isolation Forest Scores on Held-Out Benchmark Records")
    plt.xlabel("Decision score (lower values are more anomalous)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "anomaly_score_distribution.png", dpi=150)
    plt.close()

    # Graph 2: compare classifier and anomaly flags.
    comparison = pd.crosstab(
        benchmark_results["predicted_label"].map({0: "Predicted benign", 1: "Predicted attack"}),
        benchmark_results["is_anomaly"].map({False: "Not anomalous", True: "Anomalous"}),
    )
    plt.figure(figsize=(7, 5))
    sns.heatmap(comparison, annot=True, fmt="d", cmap="Blues")
    plt.title("Classifier Prediction vs. Anomaly Flag")
    plt.xlabel("Isolation Forest result")
    plt.ylabel("Random Forest result")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "classifier_anomaly_comparison.png", dpi=150)
    plt.close()

    print("Simulation outputs created:")
    for filename in [
        "simulation_benchmark_results.csv",
        "simulated_event_metadata.csv",
        "anomaly_score_distribution.png",
        "classifier_anomaly_comparison.png",
    ]:
        print(f"- outputs/{filename}")
    print(
        "\nNote: benchmark records are offline dataset examples. "
        "Synthetic event metadata is fictional and was not passed to the network model."
    )


if __name__ == "__main__":
    main()
