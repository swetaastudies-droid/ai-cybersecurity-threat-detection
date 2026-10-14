import pandas as pd
import pytest
from src.alert_engine import create_alerts


def test_alert_threshold():
    df = pd.DataFrame({
        "predicted_label": [0, 1],
        "attack_probability": [0.2, 0.8],
    })
    result = create_alerts(df, threshold=0.5)
    assert result["alert_triggered"].tolist() == [False, True]


def test_invalid_threshold():
    df = pd.DataFrame({
        "predicted_label": [1],
        "attack_probability": [0.8],
    })
    with pytest.raises(ValueError):
        create_alerts(df, threshold=1.5)
