<img width="5587" height="3057" alt="Google Sheet to ML Model-2026-10-30-161559" src="https://github.com/user-attachments/assets/259e6124-0687-4e5e-8841-439c034f479e" />

# Health ML Tracker

A personal full-stack ML application that predicts daily tiredness from self-logged lifestyle metrics.

## What it does

Each day's lifestyle data (meals, exercise, sleep, stress, etc.) is logged in a Google Sheet. A nightly pipeline pulls that data, retrains four regression models, and predicts the next day's tiredness score. Predictions and actuals are displayed in a web dashboard.

**Prediction design**: models are trained on all data up to day D and predict day D+1 — the model never sees the day it is predicting. This is walk-forward validation by construction, not a retrospective split.

## Model performance

Evaluated on all days where a next-day prediction was made and the actual score was subsequently recorded. Because the model is retrained on data up to day D before predicting D+1, every entry below is a genuine out-of-sample prediction.

| Model | Out-of-sample MAE | Notes |
|---|---|---|
| Random Forest | 0.55 | |
| Gradient Boosting | 0.50 | |
| Ridge | 0.67 | linear baseline |
| Extra Trees | 0.47 | |
| **Ensemble (mean)** | **0.53** | average of the four models |

MAE is on a 1–10 tiredness scale — a value of 1.0 means predictions are off by one point on average. Cross-validation R² for the best model: X.XX.

Hyperparameters are tuned automatically via grid search (see [ml/tune.py](ml/tune.py)) whenever 7+ new labelled days are available, with the champion model retained only if it improves on the previous CV MAE.

## Stack

| Layer | Technology |
|---|---|
| Data input | Google Sheets |
| Orchestration | Apache Airflow 2.9 (nightly DAG) |
| Database | PostgreSQL 15 |
| ML | scikit-learn — Random Forest, Gradient Boosting, Ridge, Extra Trees |
| Experiment tracking | MLflow 2.15 |
| Backend API | FastAPI + SQLAlchemy |
| Frontend | React + Recharts |
| Infrastructure | GCP Compute Engine e2-small, Docker Compose |
| CI/CD | GitHub Actions (test → deploy via SSH) |

## Project structure

```
airflow/dags/       — nightly pipeline (fetch → log actuals → check → train+infer → tune)
ml/
  features.py       — 44-column feature engineering (10-day rolling window + tiredness lag-1)
  train.py          — model training, reads best_params.json if present, saves models.pkl
  predict.py        — inference, writes to predictions table
  tune.py           — grid search, champion/challenger comparison, writes best_params.json
backend/app/        — FastAPI routes (dashboard, correlations, history)
frontend/src/       — React dashboard (trend chart, log table, correlations)
scripts/
  init_postgres.sh  — creates DBs and tables on first postgres start
tests/              — pytest suite (ML unit tests, DAG smoke tests, DB schema tests)
```

## Nightly pipeline

```
fetch_and_upsert
  ├── log_actuals          (log actual-vs-predicted errors to MLflow for closed predictions)
  └── check_new_data       (ShortCircuit: skip if all dates already have next-day predictions)
        └── train_and_infer    (retrain + predict for each pending date, oldest first)
              └── should_tune  (ShortCircuit: skip if fewer than 7 new samples since last tune)
                    └── tune_hyperparams   (grid search, update best_params.json if improved)
```

`check_new_data` short-circuits if all scored dates already have a next-day prediction.
`should_tune` short-circuits if fewer than 7 new labelled days exist since the last grid search, so tuning only runs when there is enough new signal to learn from.

## MLflow experiment tracking

Each pipeline run logs to MLflow under the `health_tiredness_prediction` experiment:

| Run type | What is logged |
|---|---|
| `train_*` | CV MAE ± std, R², ensemble CV MAE, training date range, sample count |
| `predict_*` | Per-model predictions, tagged with the training run ID |
| `actual_*` | Actual tiredness, per-model absolute and signed error (logged once per date) |
| `grid_search_*` | Best params per model, MAE delta vs champion, total fits, elapsed time |

Access the MLflow UI via SSH tunnel:

```bash
ssh -i ~/.ssh/health_ml_vm -L 15000:localhost:5000 user@<vm-ip> -N
# then open http://localhost:15000
```

## Running locally

Requires Docker and a `.env` file (see below).

```bash
docker compose up -d
```

- Airflow UI: http://localhost:8080
- Backend API: http://localhost:8000/docs
- Frontend: http://localhost:80
- MLflow UI: http://localhost:5000

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
NGINX_USER=             # username for the frontend basic auth prompt
NGINX_PASSWORD=         # password for the frontend basic auth prompt
```

## CI/CD

Pushing to `main` triggers GitHub Actions:
1. **test** — spins up postgres, runs `pytest tests/`
2. **deploy** — SSHes into the GCP VM, pulls latest code, rebuilds changed containers

Required GitHub secrets: `GCP_VM_HOST`, `GCP_VM_USER`, `GCP_SSH_PRIVATE_KEY`.
