"""Generate clearly labeled fictional events for safe UI demonstrations."""
from datetime import datetime, timezone
import pandas as pd


def generate_simulated_events(count: int = 20) -> pd.DataFrame:
    """Return synthetic demonstration metadata, not model-ready network features."""
    if count < 1 or count > 1000:
        raise ValueError("count must be between 1 and 1000.")

    scenarios = [
        ("normal_activity", "SIM-HOST-01", "routine"),
        ("failed_login_burst", "SIM-HOST-02", "multiple failed logins"),
        ("reconnaissance", "SIM-HOST-03", "port-scan-like activity"),
        ("traffic_spike", "SIM-HOST-04", "unusual traffic volume"),
    ]
    rows = []
    for i in range(count):
        scenario, host, description = scenarios[i % len(scenarios)]
        rows.append({
            "event_id": f"SIM-{i + 1:04d}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": "SIMULATED",
            "scenario": scenario,
            "source_host": host,
            "description": description,
        })
    return pd.DataFrame(rows)
