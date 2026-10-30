"""
DAG: health_pipeline
Schedule: midnight UTC every day

Tasks:
  1. fetch_and_upsert   — pull Google Sheet → upsert health_logs
  2. log_actuals        — for each date with both a prediction and an actual
                          tiredness score, log prediction errors to MLflow
                          (runs in parallel with check_new_data)
  3. check_new_data     — ShortCircuitOperator: passes only when there are
                          scored dates without a next-day prediction
  4. train_and_infer    — for each pending scored date (chronologically):
                            a. retrain on all scored rows up to that date
                            b. predict the following day
  5. should_tune        — ShortCircuitOperator: passes only when 7+ new
                          labelled samples exist since the last tuning run
  6. tune_hyperparams   — grid search over all models; writes best_params.json
                          if any model improves on champion CV MAE

If all next-day predictions are already up to date, tasks 3–6 are skipped
gracefully. If fewer than 7 new samples exist since last tuning, tasks 5–6
are skipped.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

sys.path.insert(0, '/opt/airflow')

DB_URL              = os.environ.get('APP_DB_URL')
SHEET_ID            = os.environ.get('GOOGLE_SHEET_ID')
MODEL_DIR           = os.environ.get('MODEL_DIR', '/models')
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_EXPERIMENT   = 'health_tiredness_prediction'

COLUMN_ALIASES = {
    'date':                         'log_date',
    'tiredness/dizziness':          'tiredness_score',
    'tiredness_dizziness':          'tiredness_score',
    'tiredness':                    'tiredness_score',
    'dizziness':                    'tiredness_score',
    'number_of_meals':              'num_meals',
    'meals':                        'num_meals',
    'junk':                         'num_junkfood',
    'junk_food':                    'num_junkfood',
    'num_junkfood_meals':           'num_junkfood',
    'number_of_junk_food_meals':    'num_junkfood',
    'junk_food_meals':              'num_junkfood',
    'meal_size_lunch':              'meal_size_lunch',
    'lunch_size':                   'meal_size_lunch',
    'meal_size_dinner':             'meal_size_dinner',
    'dinner_size':                  'meal_size_dinner',
    'sugar_intake':                 'sugar',
    'gym':                          'exercise_gym',
    'exercise_(gym)':               'exercise_gym',
    'excercise_gym':                'exercise_gym',
    'morning_exercise':             'exercise_morning',
    'exercise_(morning)':           'exercise_morning',
    'excercise_morning':            'exercise_morning',
}

DB_COLUMNS = {
    'log_date', 'weather', 'temperature', 'num_meals', 'num_junkfood',
    'meal_size_lunch', 'meal_size_dinner', 'sugar',
    'stomach_issues', 'sleep_quality', 'exercise_gym', 'exercise_morning',
    'stress', 'social_activity', 'social_conflicts', 'tiredness_score',
}

_PENDING_QUERY = """
    SELECT h.log_date
    FROM health_logs h
    WHERE h.tiredness_score IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM predictions p
        WHERE p.log_date = h.log_date + INTERVAL '1 day'
    )
    ORDER BY h.log_date ASC
    LIMIT 60
"""


# ── Task 1 ────────────────────────────────────────────────────────────────────

def fetch_and_upsert(**context):
    import gspread
    import google.auth
    import pandas as pd
    import sqlalchemy
    from sqlalchemy import text

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly',
    ]
    creds, _ = google.auth.default(scopes=scopes)
    gc = gspread.authorize(creds)

    worksheet = gc.open_by_key(SHEET_ID).worksheet('Blad1')
    records = worksheet.get_all_records(default_blank=None)

    if not records:
        print("Sheet is empty — nothing to fetch.")
        return

    df = pd.DataFrame(records)
    df.columns = [
        c.lower().strip().replace(' ', '_').replace('/', '_')
        for c in df.columns
    ]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    df = df[[c for c in df.columns if c in DB_COLUMNS]]

    df['log_date'] = pd.to_datetime(df['log_date'], errors='coerce').dt.date
    df = df.dropna(subset=['log_date'])

    for col in ('exercise_gym', 'exercise_morning'):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: True if str(x).strip().lower() in ('yes', '1', 'true', 'y') else False
            )

    for col in ('temperature', 'num_meals', 'num_junkfood', 'meal_size_lunch',
                'meal_size_dinner', 'sugar', 'stomach_issues', 'sleep_quality',
                'stress', 'social_activity', 'social_conflicts', 'tiredness_score'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    engine = sqlalchemy.create_engine(DB_URL)
    upserted = 0

    for _, row in df.iterrows():
        record = {k: (None if (v is None or (isinstance(v, float) and pd.isna(v))) else v)
                  for k, v in row.items()}
        record['log_date'] = str(record['log_date'])

        cols = list(record.keys())
        placeholders = ', '.join(f':{c}' for c in cols)
        updates = ', '.join(f'{c} = EXCLUDED.{c}' for c in cols if c != 'log_date')

        with engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO health_logs ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (log_date) DO UPDATE
                    SET {updates}, updated_at = NOW()
            """), record)
        upserted += 1

    print(f"Upserted {upserted} rows into health_logs.")


# ── Task 2 ────────────────────────────────────────────────────────────────────

def log_actuals(**context):
    """
    For each date where both a prediction and an actual tiredness score exist,
    log prediction errors to MLflow. Tracks logged dates in
    MODEL_DIR/actuals_logged.json so entries are written only once.
    """
    import json
    import sqlalchemy
    from sqlalchemy import text

    logged_path  = os.path.join(MODEL_DIR, 'actuals_logged.json')
    already_logged = set()
    if os.path.exists(logged_path):
        with open(logged_path) as f:
            already_logged = set(json.load(f))

    engine = sqlalchemy.create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT p.log_date,
                   p.pred_random_forest, p.pred_gradient_boost,
                   p.pred_ridge, p.pred_extra_trees, p.pred_ensemble,
                   h.tiredness_score
            FROM predictions p
            JOIN health_logs h ON h.log_date = p.log_date
            WHERE h.tiredness_score IS NOT NULL
            ORDER BY p.log_date
        """))
        rows = result.fetchall()

    new_logged  = set(already_logged)
    new_entries = 0

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        for row in rows:
            log_date_str = str(row[0])
            if log_date_str in already_logged:
                continue

            actual      = float(row[6])
            model_preds = {
                'random_forest':  row[1],
                'gradient_boost': row[2],
                'ridge':          row[3],
                'extra_trees':    row[4],
                'ensemble':       row[5],
            }

            with mlflow.start_run(run_name=f"actual_{log_date_str}"):
                mlflow.log_param('log_date', log_date_str)
                mlflow.log_metric('actual_tiredness', actual)
                for name, pred in model_preds.items():
                    if pred is not None:
                        pred = float(pred)
                        mlflow.log_metric(f'pred_{name}',         pred)
                        mlflow.log_metric(f'abs_error_{name}',    abs(actual - pred))
                        mlflow.log_metric(f'signed_error_{name}', actual - pred)

            new_logged.add(log_date_str)
            new_entries += 1

    except Exception as e:
        print(f"MLflow actuals log failed (non-fatal): {e}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(logged_path, 'w') as f:
        json.dump(sorted(new_logged), f)

    print(f"Logged {new_entries} new actuals to MLflow.")


# ── Task 3 ────────────────────────────────────────────────────────────────────

def check_new_data(**context) -> bool:
    """
    Returns True only when there are scored dates that don't yet have a
    next-day prediction. No Airflow Variable needed — the DB is the source
    of truth.
    """
    import sqlalchemy
    from sqlalchemy import text

    engine = sqlalchemy.create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text(
            f"SELECT COUNT(*) FROM ({_PENDING_QUERY}) pending"
        ))
        pending_count = result.scalar()

    if pending_count > 0:
        print(f"{pending_count} date(s) need next-day predictions. Proceeding.")
        return True

    print("All next-day predictions are up to date. Skipping.")
    return False


# ── Task 4 ────────────────────────────────────────────────────────────────────

def train_and_infer(**context):
    """
    For each pending scored date (oldest first):
      1. Retrain all models on scored rows up to and including that date.
      2. Predict the following day using the freshly trained model.

    This guarantees every prediction is made blind — the model never sees
    the date it is predicting.
    """
    import sqlalchemy
    from sqlalchemy import text
    from ml.train import train_models
    from ml.predict import predict_for_date

    engine = sqlalchemy.create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text(_PENDING_QUERY))
        pending = [row[0] for row in result]

    if not pending:
        print("No pending dates found.")
        return

    print(f"Processing {len(pending)} pending date(s).")
    for scored_date in pending:
        next_day = scored_date + timedelta(days=1)
        print(f"Training on data up to {scored_date}, predicting {next_day}.")
        train_result = train_models(DB_URL, MODEL_DIR, up_to_date=scored_date)
        print(f"  Train result: {train_result}")
        predict_for_date(next_day, DB_URL, MODEL_DIR,
                         mlflow_run_id=train_result.get('mlflow_run_id'))
        print(f"  Predicted {next_day}.")


# ── Task 5 ────────────────────────────────────────────────────────────────────

def should_tune(**context) -> bool:
    """
    Returns True only when 7+ new labelled samples exist since the last
    grid search run (tracked via MODEL_DIR/last_tuned.txt).
    """
    import sqlalchemy
    from sqlalchemy import text

    last_tuned_path = os.path.join(MODEL_DIR, 'last_tuned.txt')
    last_tuned = None
    if os.path.exists(last_tuned_path):
        with open(last_tuned_path) as f:
            last_tuned = f.read().strip()

    engine = sqlalchemy.create_engine(DB_URL)
    with engine.connect() as conn:
        if last_tuned:
            result = conn.execute(text("""
                SELECT COUNT(*) FROM health_logs
                WHERE tiredness_score IS NOT NULL AND log_date > :last_tuned
            """), {'last_tuned': last_tuned})
        else:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM health_logs WHERE tiredness_score IS NOT NULL"
            ))
        count = result.scalar()

    if count >= 7:
        print(f"{count} new sample(s) since last tuning. Proceeding with grid search.")
        return True
    print(f"Only {count} new sample(s) since last tuning (need 7). Skipping grid search.")
    return False


# ── Task 6 ────────────────────────────────────────────────────────────────────

def tune_hyperparams(**context):
    from ml.tune import run_grid_search
    result = run_grid_search(DB_URL, MODEL_DIR)
    print(f"Grid search complete: {result}")


# ── DAG definition ─────────────────────────────────────────────────────────────

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='health_pipeline',
    default_args=default_args,
    description='Nightly health data pipeline: fetch → check → train+infer → tune',
    schedule_interval='0 0 * * *',
    catchup=False,
    tags=['health'],
) as dag:

    t_fetch = PythonOperator(
        task_id='fetch_and_upsert',
        python_callable=fetch_and_upsert,
    )

    t_log_actuals = PythonOperator(
        task_id='log_actuals',
        python_callable=log_actuals,
    )

    t_check = ShortCircuitOperator(
        task_id='check_new_data',
        python_callable=check_new_data,
    )

    t_train_and_infer = PythonOperator(
        task_id='train_and_infer',
        python_callable=train_and_infer,
    )

    t_should_tune = ShortCircuitOperator(
        task_id='should_tune',
        python_callable=should_tune,
    )

    t_tune = PythonOperator(
        task_id='tune_hyperparams',
        python_callable=tune_hyperparams,
    )

    t_fetch >> t_log_actuals
    t_fetch >> t_check >> t_train_and_infer >> t_should_tune >> t_tune
