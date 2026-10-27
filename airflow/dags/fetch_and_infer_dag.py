"""
DAG: health_pipeline
Schedule: midnight UTC every day

Tasks:
  1. fetch_and_upsert   — pull Google Sheet → upsert health_logs
  2. check_new_data     — ShortCircuitOperator: passes only when there are
                          scored dates without a next-day prediction
  3. train_and_infer    — for each pending scored date (chronologically):
                            a. retrain on all scored rows up to that date
                            b. predict the following day
                          This ensures each prediction is made without the
                          model having seen the date being predicted.

If all next-day predictions are already up to date, tasks 2 and 3 are
skipped gracefully.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

sys.path.insert(0, '/opt/airflow')

DB_URL    = os.environ.get('APP_DB_URL')
SHEET_ID  = os.environ.get('GOOGLE_SHEET_ID')
MODEL_DIR = os.environ.get('MODEL_DIR', '/models')

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


# ── Task 3 ────────────────────────────────────────────────────────────────────

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
    description='Nightly health data pipeline: fetch → check → train+infer',
    schedule_interval='0 0 * * *',
    catchup=False,
    tags=['health'],
) as dag:

    t_fetch = PythonOperator(
        task_id='fetch_and_upsert',
        python_callable=fetch_and_upsert,
    )

    t_check = ShortCircuitOperator(
        task_id='check_new_data',
        python_callable=check_new_data,
    )

    t_train_and_infer = PythonOperator(
        task_id='train_and_infer',
        python_callable=train_and_infer,
    )

    t_fetch >> t_check >> t_train_and_infer
