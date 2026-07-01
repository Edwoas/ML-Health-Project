"""
Verify that init_postgres.sh created the expected schema.
Requires TEST_DB_URL to point at a real Postgres instance.
"""

import os
import pytest
import psycopg2

DB_URL = os.environ.get("TEST_DB_URL", "postgresql://health:testpass@localhost:5432/health")


@pytest.fixture(scope="module")
def conn():
    c = psycopg2.connect(DB_URL)
    yield c
    c.close()


def _table_exists(conn, table):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.%s')" % table)
        return cur.fetchone()[0] is not None


def _columns(conn, table):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        )
        return {row[0] for row in cur.fetchall()}


def test_health_logs_exists(conn):
    assert _table_exists(conn, "health_logs")


def test_predictions_exists(conn):
    assert _table_exists(conn, "predictions")


def test_prediction_history_exists(conn):
    assert _table_exists(conn, "prediction_history")


def test_health_logs_columns(conn):
    cols = _columns(conn, "health_logs")
    required = {
        "log_date", "tiredness_score", "num_meals", "num_junkfood",
        "meal_size_lunch", "meal_size_dinner", "sugar", "sleep_quality",
        "stress", "social_activity", "social_conflicts",
        "exercise_gym", "exercise_morning", "stomach_issues",
    }
    assert required <= cols, f"Missing columns: {required - cols}"


def test_predictions_columns(conn):
    cols = _columns(conn, "predictions")
    required = {
        "log_date", "pred_random_forest", "pred_gradient_boost",
        "pred_ridge", "pred_extra_trees", "pred_ensemble",
    }
    assert required <= cols, f"Missing columns: {required - cols}"


def test_prediction_history_columns(conn):
    cols = _columns(conn, "prediction_history")
    required = {
        "log_date", "pred_ensemble", "run_at",
        "pred_random_forest", "pred_gradient_boost",
        "pred_ridge", "pred_extra_trees",
    }
    assert required <= cols, f"Missing columns: {required - cols}"


def test_health_logs_log_date_is_unique(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'health_logs'
            AND indexdef ILIKE '%unique%'
            AND indexdef ILIKE '%log_date%'
        """)
        assert cur.fetchone()[0] >= 1


def test_prediction_history_log_date_is_not_unique(conn):
    """prediction_history is append-only — must NOT have unique constraint on log_date."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'prediction_history'
            AND indexdef ILIKE '%unique%'
            AND indexdef ILIKE '%log_date%'
        """)
        assert cur.fetchone()[0] == 0


def test_can_insert_and_upsert_health_log(conn):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO health_logs (log_date, tiredness_score)
            VALUES ('1900-01-01', 7)
            ON CONFLICT (log_date) DO UPDATE SET tiredness_score = EXCLUDED.tiredness_score
        """)
        cur.execute("SELECT tiredness_score FROM health_logs WHERE log_date = '1900-01-01'")
        assert cur.fetchone()[0] == 7
        cur.execute("DELETE FROM health_logs WHERE log_date = '1900-01-01'")
    conn.commit()
