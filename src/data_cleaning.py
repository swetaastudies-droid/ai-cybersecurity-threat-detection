"""Basic data-quality checks; fitted imputation happens in the ML pipeline."""
import numpy as np
import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and replace numeric infinities with NaN."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    numeric = cleaned.select_dtypes(include=["number"]).columns
    cleaned[numeric] = cleaned[numeric].replace([np.inf, -np.inf], np.nan)

    # Do not silently invent labels for rows without ground truth.
    if "label" in cleaned.columns:
        if cleaned["label"].isna().any():
            raise ValueError("Target label contains missing values.")
        cleaned["label"] = pd.to_numeric(cleaned["label"], errors="raise").astype(int)
        if not set(cleaned["label"].unique()).issubset({0, 1}):
            raise ValueError("Expected binary labels 0 (benign) and 1 (attack).")

    return cleaned


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact report for inspecting data quality."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique(dropna=False),
    })
