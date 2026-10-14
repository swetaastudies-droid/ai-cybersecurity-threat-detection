import pytest
from src.simulation import generate_simulated_events


def test_events_are_marked_simulated():
    df = generate_simulated_events(8)
    assert len(df) == 8
    assert set(df["environment"]) == {"SIMULATED"}


def test_invalid_count():
    with pytest.raises(ValueError):
        generate_simulated_events(0)
