"""Optional Isolation Forest anomaly detector."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "anomaly_detection_model.joblib"


def train_anomaly_detector(X_train: pd.DataFrame, y_train: pd.Series, preprocessor):
    """Fit Isolation Forest on benign training rows only.

    This is an optional unsupervised signal. It is not an attack probability.
    """
    benign = X_train.loc[y_train == 0]
    if benign.empty:
        raise ValueError("No benign training rows available for anomaly detector.")

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("detector", IsolationForest(
            n_estimators=150,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(benign)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def score_anomalies(records: pd.DataFrame) -> pd.DataFrame:
    """Return anomaly flag and decision score; lower scores are more anomalous."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Anomaly model missing. Run python train_model.py.")
    model = joblib.load(MODEL_PATH)
    # IsolationForest predicts -1 for anomaly and 1 for inlier.
    raw_prediction = model.predict(records)
    score = model.decision_function(records)
    return pd.DataFrame({
        "is_anomaly": raw_prediction == -1,
        "anomaly_score": score,
    }, index=records.index)
