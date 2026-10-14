"""Load and validate the UNSW-NB15 training and testing CSV files."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and perform basic schema checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download the UNSW-NB15 training/testing CSVs and place them in data/raw/."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


def load_train_test():
    """Load the official train/test files and validate binary labels."""
    train = load_csv(TRAIN_PATH)
    test = load_csv(TEST_PATH)

    for name, frame in (("training", train), ("testing", test)):
        if "label" not in frame.columns:
            raise ValueError(f"The {name} file has no 'label' column.")
        if frame["label"].isna().any():
            raise ValueError(f"The {name} file contains missing labels.")
        labels = pd.to_numeric(frame["label"], errors="raise")
        if not set(labels.unique()).issubset({0, 1}):
            raise ValueError(f"The {name} file must use labels 0 and 1.")
        frame["label"] = labels.astype(int)

    return train, test


if __name__ == "__main__":
    train, test = load_train_test()
    print(f"Training rows/columns: {train.shape}")
    print(f"Testing rows/columns:  {test.shape}")
    print("\nTraining labels:\n", train["label"].value_counts().sort_index())
    print("\nTesting labels:\n", test["label"].value_counts().sort_index())
