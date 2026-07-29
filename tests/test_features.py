from __future__ import annotations

import json

import pytest

from air_quality.features import FEATURES, build_dataset


@pytest.fixture
def sample_raw(tmp_path):
    """Create a minimal raw JSON file for testing."""
    data = {
        "metadata": {"latitude": 24.7136, "longitude": 46.6753},
        "air_quality": {
            "hourly": {
                "time": [
                    "2026-04-25T00:00", "2026-04-25T01:00",
                    "2026-04-25T02:00", "2026-04-25T03:00",
                    "2026-04-25T04:00", "2026-04-25T05:00",
                    "2026-04-25T06:00", "2026-04-25T07:00",
                    "2026-04-25T08:00", "2026-04-25T09:00",
                    "2026-04-25T10:00",
                ],
                "pm2_5": [15, 18, 22, 20, 25, 30, 35, 40, 38, 33, 28],
                "pm10": [30, 35, 40, 38, 45, 55, 60, 70, 65, 55, 50],
            }
        },
        "weather": {
            "hourly": {
                "time": [
                    "2026-04-25T00:00", "2026-04-25T01:00",
                    "2026-04-25T02:00", "2026-04-25T03:00",
                    "2026-04-25T04:00", "2026-04-25T05:00",
                    "2026-04-25T06:00", "2026-04-25T07:00",
                    "2026-04-25T08:00", "2026-04-25T09:00",
                    "2026-04-25T10:00",
                ],
                "temperature_2m": [28, 27, 26, 26, 27, 29, 31, 33, 35, 36, 36],
                "relative_humidity_2m": [40, 42, 44, 43, 38, 32, 25, 20, 18, 17, 16],
                "wind_speed_10m": [12, 10, 8, 9, 11, 15, 18, 20, 22, 21, 18],
            }
        },
    }
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_build_dataset_creates_expected_columns(sample_raw, tmp_path):
    output = tmp_path / "output.parquet"
    df = build_dataset(sample_raw, str(output), threshold=35.0)
    for col in FEATURES + ["high_pollution_next_hour", "time"]:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0


def test_build_dataset_target_values(sample_raw, tmp_path):
    output = tmp_path / "output.parquet"
    df = build_dataset(sample_raw, str(output), threshold=100.0)
    assert df["high_pollution_next_hour"].isin([0, 1]).all()
    # After shift + dropna, data starts at row where rolling window is populated (row 6 of original)
    # Original row 6 has pm2_5=35 at 06:00, next hour (07:00) pm2_5=40 < 100 => target 0
    assert df["high_pollution_next_hour"].iloc[0] == 0
    # Last original row with a valid target (row 9, pm2_5=33 at 09:00, next=28 < 100 => target 0)
    assert df["high_pollution_next_hour"].iloc[-1] == 0


def test_no_future_leakage(sample_raw, tmp_path):
    """Verify lag/rolling features use shift() and don't peek into the future."""
    output = tmp_path / "output.parquet"
    df = build_dataset(sample_raw, str(output), threshold=35.0)
    # pm2_5_lag_1 at row i should equal pm2_5 at row i-1
    for i in range(1, len(df)):
        assert df["pm2_5_lag_1"].iloc[i] == df["pm2_5"].iloc[i - 1], (
            f"Lag 1 leak at row {i}: {df['pm2_5_lag_1'].iloc[i]} != {df['pm2_5'].iloc[i - 1]}"
        )
