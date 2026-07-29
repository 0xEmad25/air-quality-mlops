from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "threshold" in data
    assert "features" in data
    assert data["version"] == "1.0.0"


def test_valid_prediction() -> None:
    payload = {
        "pm2_5": 25.0,
        "pm10": 60.0,
        "temperature_2m": 32.0,
        "relative_humidity_2m": 25.0,
        "wind_speed_10m": 12.0,
        "hour": 12,
        "day_of_week": 2,
        "pm2_5_lag_1": 24.0,
        "pm2_5_lag_3": 22.0,
        "pm2_5_rolling_mean_6": 23.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "risk_level" in data
    assert data["model_version"] == "1.0.0"


def test_invalid_prediction_payload() -> None:
    response = client.post("/predict", json={"pm2_5": 10})
    assert response.status_code == 422


def test_invalid_field_range() -> None:
    """Hour must be 0-23, so 25 should fail validation."""
    payload = {
        "pm2_5": 25.0,
        "pm10": 60.0,
        "temperature_2m": 32.0,
        "relative_humidity_2m": 25.0,
        "wind_speed_10m": 12.0,
        "hour": 25,
        "day_of_week": 2,
        "pm2_5_lag_1": 24.0,
        "pm2_5_lag_3": 22.0,
        "pm2_5_rolling_mean_6": 23.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
