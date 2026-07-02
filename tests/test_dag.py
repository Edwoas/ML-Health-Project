"""
Smoke tests for the Airflow DAG file — no Airflow installation required.
"""

import ast
import os

DAG_PATH = os.path.join(os.path.dirname(__file__), "..", "airflow", "dags", "fetch_and_infer_dag.py")


def _source():
    with open(DAG_PATH) as f:
        return f.read()


def test_dag_is_valid_python():
    ast.parse(_source())


def test_dag_has_all_three_tasks():
    src = _source()
    for task_id in ("fetch_and_upsert", "check_new_data", "train_and_infer"):
        assert task_id in src, f"Task '{task_id}' not found in DAG"


def test_dag_old_tasks_removed():
    src = _source()
    assert "task_id='train_models'" not in src
    assert "task_id='run_inference'" not in src


def test_dag_uses_short_circuit_operator():
    assert "ShortCircuitOperator" in _source()


def test_dag_uses_adc_not_key_file():
    src = _source()
    assert "google.auth.default" in src, "DAG must use Application Default Credentials"
    assert "from_service_account_file" not in src, "DAG must not use SA key file auth"


def test_dag_schedule_is_midnight():
    assert "0 0 * * *" in _source()


def test_dag_task_chain():
    src = _source()
    chain = "t_fetch >> t_check >> t_train_and_infer"
    assert chain in src, f"Expected task chain '{chain}' not found"


def test_dag_no_airflow_variable():
    """check_new_data must not use Airflow Variables — DB is source of truth."""
    src = _source()
    assert "Variable.get" not in src
    assert "Variable.set" not in src
    assert "Variable.delete" not in src


def test_dag_pending_query_excludes_target_date():
    """Predictions must be for log_date + 1 day, never for the scored date itself."""
    src = _source()
    assert "log_date + INTERVAL '1 day'" in src


def test_dag_uses_up_to_date():
    """train_and_infer must pass up_to_date to train_models."""
    assert "up_to_date=scored_date" in _source()
