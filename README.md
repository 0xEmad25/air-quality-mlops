# Riyadh Air Quality Intelligence Platform

Predict whether the next hour in Riyadh will have high PM2.5 pollution risk.

## Project Structure

```
air-quality-mlops/
├── app/
│   ├── api/main.py          # FastAPI prediction service
│   └── frontend/app.py      # Streamlit user interface
├── src/air_quality/
│   ├── collect.py            # API data collection
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training + MLflow
│   └── monitoring.py         # Evidently drift report
├── tests/
│   ├── test_features.py      # Feature pipeline tests
│   └── test_api.py           # API endpoint tests
├── data/
│   ├── raw/                  # Immutable raw API data
│   ├── processed/            # Feature-engineered parquet
│   └── monitoring/           # Prediction logs + reference
├── models/                   # Trained model artifacts
├── reports/                  # Evidently HTML reports
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── .env.example
```

## Quick Start

```bash
# Install dependencies
uv sync

# Collect data (90 days of Riyadh hourly data)
uv run python -m air_quality.collect

# Build features
uv run python -m air_quality.features

# Train models (3 candidates logged to MLflow)
# First start MLflow server in another terminal:
#   uv run mlflow server --host 0.0.0.0 --port 5001
uv run python -m air_quality.train

# Start API
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
API_URL=http://localhost:8000 uv run streamlit run app/frontend/app.py --server.address 0.0.0.0 --server.port 8501

# Generate monitoring report (after sending predictions)
uv run python -m air_quality.monitoring
```

## Quality Gate

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

## API Endpoints

- `GET /health` — service status
- `GET /model-info` — model metadata + feature list
- `POST /predict` — prediction request (10 features)

## Docker

```bash
docker compose up --build -d
curl http://localhost:8000/health
```

## Deployment (Dokploy)

1. Push this repo to GitHub
2. Create a Docker Compose project in Dokploy
3. Connect the repository and branch
4. Set the Compose file path to `docker-compose.yml`
5. Assign domains to the API (port 8000) and frontend (port 8501) services
6. Deploy and verify with the health endpoint

## Model Summary

| Metric | Validation | Test |
|--------|-----------|------|
| Selected model | Logistic Regression | Logistic Regression |
| F1-score | 0.9379 | 0.8916 |
| Recall (high-risk) | 0.9067 | 0.8268 |
| ROC-AUC | 0.9812 | 0.9761 |
| Threshold | 0.50 | 0.50 |

**Note:** The PM2.5 threshold for defining "high pollution" was raised from 35 to 100 µg/m³ to create a more balanced class distribution (65% high / 35% normal in the training data). This makes the model's predictions meaningful — clean air (PM2.5≈15) shows 0.5% risk, while severe pollution (PM2.5≈200) shows 98.7% risk.

## Data Source

- Air quality: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- Weather: [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api)
- Location: Riyadh (24.7136°N, 46.6753°E)
- Period: 90 days of hourly observations
