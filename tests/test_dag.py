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


def test_dag_has_all_four_tasks():
    src = _source()
    for task_id in ("fetch_and_upsert", "check_new_data", "train_models", "run_inference"):
        assert task_id in src, f"Task '{task_id}' not found in DAG"


def test_dag_uses_short_circuit_operator():
    assert "ShortCircuitOperator" in _source()


def test_dag_uses_adc_not_key_file():
    src = _source()
    assert "google.auth.default" in src, "DAG must use Application Default Credentials"
    assert "from_service_account_file" not in src, "DAG must not use SA key file auth"


def test_dag_schedule_is_midnight():
    assert "0 0 * * *" in _source()


def test_dag_task_chain():
    """The four tasks must be chained in the correct order."""
    src = _source()
    chain = "t_fetch >> t_check >> t_train >> t_infer"
    assert chain in src, f"Expected task chain '{chain}' not found"
