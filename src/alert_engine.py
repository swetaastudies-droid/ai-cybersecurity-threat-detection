"""Simple explainable triage rules for model outputs."""
import pandas as pd


def create_alerts(predictions: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """Mark records for review using an adjustable model-score threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    required = {"predicted_label", "attack_probability"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions must include: {sorted(required)}")

    result = predictions.copy()
    result["alert_triggered"] = result["attack_probability"] >= threshold
    result["severity"] = result["attack_probability"].apply(
        lambda p: "High" if p >= 0.90 else
                  "Medium" if p >= 0.70 else
                  "Low" if p >= 0.50 else "Informational"
    )
    result["investigation_status"] = "Needs analyst review"
    return result
