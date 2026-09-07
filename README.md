# Health ML Tracker

A personal full-stack ML application that predicts daily tiredness from self-logged lifestyle metrics.

## What it does

Each day's lifestyle data (meals, exercise, sleep, stress, etc.) is logged in a Google Sheet. A nightly pipeline pulls that data, retrains four regression models, and predicts the next day's tiredness score. Predictions and actuals are displayed in a web dashboard.

**Prediction design**: models are trained on all data up to day D and predict day D+1 — the model never sees the day it is predicting.

## Stack

| Layer | Technology |
|---|---|
| Data input | Google Sheets |
| Orchestration | Apache Airflow 2.9 (nightly DAG) |
| Database | PostgreSQL 15 |
| ML | scikit-learn — Random Forest, Gradient Boosting, Ridge, Extra Trees |
| Backend API | FastAPI + SQLAlchemy |
| Frontend | React + Recharts |
| Infrastructure | GCP Compute Engine e2-small, Docker Compose |
| CI/CD | GitHub Actions (test → deploy via SSH) |

## Project structure

```
airflow/dags/       — nightly pipeline (fetch → check → train+infer)
ml/
  features.py       — 43-column feature engineering (10-day rolling window)
  train.py          — model training, saves models.pkl
  predict.py        — inference, writes to predictions table
backend/app/        — FastAPI routes (dashboard, correlations, history)
frontend/src/       — React dashboard (trend chart, log table, correlations)
scripts/
  init_postgres.sh  — creates DBs and tables on first postgres start
tests/              — pytest suite (ML unit tests, DAG smoke tests, DB schema tests)
```

## Nightly pipeline

```
fetch_and_upsert  →  check_new_data  →  train_and_infer
   (Google Sheet)     (ShortCircuit)     (retrain + predict for each pending date)
```

`check_new_data` short-circuits the pipeline if all scored dates already have a next-day prediction, so the train/infer step is skipped on nights with no new data.

## Running locally

Requires Docker and a `.env` file (see below).

```bash
docker compose up -d
```

- Airflow UI: http://localhost:8080
- Backend API: http://localhost:8000/docs
- Frontend: http://localhost:80

## Environment variables (`.env`)

```
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
AIRFLOW_FERNET_KEY=     # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW_SECRET_KEY=
AIRFLOW_ADMIN_USER=
AIRFLOW_ADMIN_PASSWORD=
AIRFLOW_UID=50000
GOOGLE_SHEET_ID=        # the ID from the Sheet URL
```

## CI/CD

Pushing to `main` triggers GitHub Actions:
1. **test** — spins up postgres, runs `pytest tests/`
2. **deploy** — SSHes into the GCP VM, pulls latest code, rebuilds changed containers

Required GitHub secrets: `GCP_VM_HOST`, `GCP_VM_USER`, `GCP_SSH_PRIVATE_KEY`.
