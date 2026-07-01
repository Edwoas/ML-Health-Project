"""
DAG: health_pipeline
Schedule: midnight UTC every day

Tasks:
  1. fetch_and_upsert  — pull Google Sheet → upsert health_logs
  2. check_new_data    — ShortCircuitOperator: passes only when new scored rows appeared
  3. train_models      — retrain all four models
  4. run_inference     — generate predictions for every date that lacks one

If no new tiredness-scored rows were added since the last successful run,
tasks 3 and 4 are skipped gracefully.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator

sys.path.insert(0, '/opt/airflow')

DB_URL    = os.environ.get('APP_DB_URL')
SHEET_ID  = os.environ.get('GOOGLE_SHEET_ID')
SA_PATH   = os.environ.get('GOOGLE_SERVICE_ACCOUNT_PATH',
                            '/opt/airflow/credentials/service-account.json')
MODEL_DIR = os.environ.get('MODEL_DIR', '/models')

# Airflow variable name that tracks how many scored rows we've seen
_SCORED_COUNT_VAR = 'health_last_scored_count'

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
    # Use Application Default Credentials — on the GCP VM this automatically
    # picks up the service account attached to the instance, no key file needed.
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
    Returns True (allow downstream tasks) only when the count of rows with a
    tiredness_score has increased since the last successful pipeline run.
    """
    import sqlalchemy
    from sqlalchemy import text

    engine = sqlalchemy.create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT COUNT(*) FROM health_logs WHERE tiredness_score IS NOT NULL"
        ))
        current_count = result.scalar()

    last_count = int(Variable.get(_SCORED_COUNT_VAR, default_var=0))

    if current_count > last_count:
        print(f"New scored rows detected: {last_count} → {current_count}. Proceeding.")
        # Save updated count so next run can compare
        Variable.set(_SCORED_COUNT_VAR, current_count)
        return True

    print(f"No new scored rows ({current_count} unchanged). Skipping training & inference.")
    return False


# ── Task 3 ────────────────────────────────────────────────────────────────────

def train(**context):
    from ml.train import train_models
    result = train_models(DB_URL, MODEL_DIR)
    print(f"Training result: {result}")


# ── Task 4 ────────────────────────────────────────────────────────────────────

def run_inference(**context):
    import sqlalchemy
    from sqlalchemy import text
    from ml.predict import predict_for_date

    engine = sqlalchemy.create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT h.log_date
            FROM health_logs h
            LEFT JOIN predictions p ON h.log_date = p.log_date
            WHERE p.log_date IS NULL
            ORDER BY h.log_date DESC
            LIMIT 60
        """))
        pending = [row[0] for row in result]

    if not pending:
        print("All logged dates already have predictions.")
        return

    print(f"Running inference for {len(pending)} date(s).")
    for d in pending:
        predict_for_date(d, DB_URL, MODEL_DIR)


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
    description='Nightly health data pipeline: fetch → check → train → infer',
    schedule_interval='0 0 * * *',   # midnight UTC
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

    t_train = PythonOperator(
        task_id='train_models',
        python_callable=train,
    )

    t_infer = PythonOperator(
        task_id='run_inference',
        python_callable=run_inference,
    )

    t_fetch >> t_check >> t_train >> t_infer
