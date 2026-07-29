"""
Sanity checks for the air quality model.
Run: uv run python test_sanity.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests

# ── Test 1: Model probability ranges ──
print("=" * 50)
print("Test 1: Model probability sanity")
print("=" * 50)

artifact = joblib.load("models/model.joblib")
pipeline = artifact["pipeline"]

scenarios = [
    ("Clean air (PM2.5=10)", {"pm2_5": 10.0, "pm10": 20.0, "temperature_2m": 25.0, "relative_humidity_2m": 40.0, "wind_speed_10m": 15.0, "hour": 14, "day_of_week": 3, "pm2_5_lag_1": 9.0, "pm2_5_lag_3": 8.0, "pm2_5_rolling_mean_6": 10.0}),
    ("Moderate (PM2.5=50)", {"pm2_5": 50.0, "pm10": 80.0, "temperature_2m": 32.0, "relative_humidity_2m": 25.0, "wind_speed_10m": 12.0, "hour": 14, "day_of_week": 3, "pm2_5_lag_1": 45.0, "pm2_5_lag_3": 40.0, "pm2_5_rolling_mean_6": 42.0}),
    ("High (PM2.5=150)", {"pm2_5": 150.0, "pm10": 250.0, "temperature_2m": 40.0, "relative_humidity_2m": 15.0, "wind_speed_10m": 5.0, "hour": 14, "day_of_week": 3, "pm2_5_lag_1": 140.0, "pm2_5_lag_3": 130.0, "pm2_5_rolling_mean_6": 135.0}),
    ("Severe (PM2.5=300)", {"pm2_5": 300.0, "pm10": 500.0, "temperature_2m": 45.0, "relative_humidity_2m": 8.0, "wind_speed_10m": 3.0, "hour": 14, "day_of_week": 3, "pm2_5_lag_1": 280.0, "pm2_5_lag_3": 260.0, "pm2_5_rolling_mean_6": 270.0}),
]

all_pass = True
for name, data in scenarios:
    frame = pd.DataFrame([data])
    prob = pipeline.predict_proba(frame)[0, 1]
    pred = "HIGH" if prob >= 0.5 else "normal"
    ok = (name.startswith("Clean") and prob < 0.2) or (name.startswith("Severe") and prob > 0.8) or (name.startswith("High") and prob > 0.5) or (name.startswith("Moderate") and prob < 0.5)
    status = "✓" if ok else "✗"
    if not ok:
        all_pass = False
    print(f"  {status} {name:30s} → {pred:>6s} ({prob:.1%})")

print()

# ── Test 2: Feature pipeline reproducibility ──
print("=" * 50)
print("Test 2: Feature pipeline produces expected columns")
print("=" * 50)
from air_quality.features import FEATURES, build_dataset
df = build_dataset("data/raw/air_quality.json", "/tmp/sanity_test.parquet")
expected_cols = set(FEATURES + ["high_pollution_next_hour", "time"])
actual_cols = set(df.columns)
if expected_cols.issubset(actual_cols):
    print(f"  ✓ All expected columns present ({len(df)} rows)")
else:
    print(f"  ✗ Missing: {expected_cols - actual_cols}")
    all_pass = False

# ── Test 3: API endpoints (if running) ──
print()
print("=" * 50)
print("Test 3: API endpoint check (start API first if wanted)")
print("=" * 50)
try:
    r = requests.get("http://localhost:8000/health", timeout=3)
    if r.status_code == 200 and r.json()["model_loaded"]:
        print("  ✓ /health → ok, model loaded")

        # valid prediction
        payload = scenarios[1][1]  # moderate scenario
        r2 = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        if r2.status_code == 200:
            data2 = r2.json()
            print(f"  ✓ /predict → prob={data2['probability']:.1%}, risk={data2['risk_level']}")
        else:
            print(f"  ✗ /predict → {r2.status_code}")
            all_pass = False

        # invalid (should be 422)
        r3 = requests.post("http://localhost:8000/predict", json={"pm2_5": 10}, timeout=3)
        if r3.status_code == 422:
            print("  ✓ Invalid payload → 422")
        else:
            print(f"  ✗ Invalid payload → {r3.status_code} (expected 422)")
            all_pass = False
    else:
        print(f"  ✗ /health → {r.json()}")
        all_pass = False
except requests.ConnectionError:
    print("  - API not running (start with: uv run uvicorn app.api.main:app ...)")
except Exception as e:
    print(f"  ✗ API error: {e}")
    all_pass = False

print()
print("=" * 50)
if all_pass:
    print("All tests passed ✓")
else:
    print("Some tests failed ✗")
    sys.exit(1)
