"""
Run inference for a given date using the four saved models.
Writes all four predictions + their mean (ensemble) to the predictions table.
"""

import os
import sys
from datetime import date

import joblib
import sqlalchemy
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_EXPERIMENT   = 'health_tiredness_prediction'


def _clip(v: float) -> float:
    return round(max(1.0, min(10.0, v)), 2)


def _log_prediction_to_mlflow(log_date, preds: dict, mlflow_run_id=None):
    """Log inference results to MLflow. Non-fatal if the server is unavailable."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        tags = {'training_run_id': mlflow_run_id} if mlflow_run_id else {}
        with mlflow.start_run(run_name=f"predict_{log_date}", tags=tags):
            mlflow.log_param('target_date', str(log_date))
            for name, val in preds.items():
                mlflow.log_metric(f'pred_{name}', val)
    except Exception as e:
        print(f"MLflow prediction log failed (non-fatal): {e}")


def predict_for_date(log_date: date, db_url: str, model_dir: str = '/models',
                     mlflow_run_id: str = None) -> dict | None:
    from ml.features import get_inference_features

    models_path = os.path.join(model_dir, 'models.pkl')
    if not os.path.exists(models_path):
        print(f"No model bundle at {models_path}. Run training first.")
        return None

    models_dict = joblib.load(models_path)

    X = get_inference_features(log_date, db_url)
    if X is None:
        print(f"No data for {log_date}. Skipping inference.")
        return None

    preds = {name: _clip(float(pipeline.predict(X)[0]))
             for name, pipeline in models_dict.items()}
    preds['ensemble'] = _clip(sum(preds.values()) / len(preds))

    # Resolve current model version
    version = None
    try:
        versioned = sorted(
            f for f in os.listdir(model_dir)
            if f.startswith('models_') and f.endswith('.pkl') and f != 'models.pkl'
        )
        if versioned:
            version = versioned[-1].replace('models_', '').replace('.pkl', '')
    except Exception:
        pass

    params = {
        'log_date':           str(log_date),
        'pred_random_forest':  preds.get('random_forest'),
        'pred_gradient_boost': preds.get('gradient_boost'),
        'pred_ridge':          preds.get('ridge'),
        'pred_extra_trees':    preds.get('extra_trees'),
        'pred_ensemble':       preds['ensemble'],
        'model_version':       version,
    }

    engine = sqlalchemy.create_engine(db_url)
    with engine.begin() as conn:
        # Upsert latest prediction (used by dashboard)
        conn.execute(text("""
            INSERT INTO predictions
                (log_date, pred_random_forest, pred_gradient_boost,
                 pred_ridge, pred_extra_trees, pred_ensemble, model_version)
            VALUES
                (:log_date, :pred_random_forest, :pred_gradient_boost,
                 :pred_ridge, :pred_extra_trees, :pred_ensemble, :model_version)
            ON CONFLICT (log_date) DO UPDATE SET
                pred_random_forest  = EXCLUDED.pred_random_forest,
                pred_gradient_boost = EXCLUDED.pred_gradient_boost,
                pred_ridge          = EXCLUDED.pred_ridge,
                pred_extra_trees    = EXCLUDED.pred_extra_trees,
                pred_ensemble       = EXCLUDED.pred_ensemble,
                model_version       = EXCLUDED.model_version,
                created_at          = NOW()
        """), params)

        # Always append to history — never overwrite, so we can trace model improvement
        conn.execute(text("""
            INSERT INTO prediction_history
                (log_date, pred_random_forest, pred_gradient_boost,
                 pred_ridge, pred_extra_trees, pred_ensemble, model_version)
            VALUES
                (:log_date, :pred_random_forest, :pred_gradient_boost,
                 :pred_ridge, :pred_extra_trees, :pred_ensemble, :model_version)
        """), params)

    print(
        f"[{log_date}] RF={preds.get('random_forest'):.2f}  "
        f"GB={preds.get('gradient_boost'):.2f}  "
        f"Ridge={preds.get('ridge'):.2f}  "
        f"ET={preds.get('extra_trees'):.2f}  "
        f"Ensemble={preds['ensemble']:.2f}"
    )
    _log_prediction_to_mlflow(log_date, preds, mlflow_run_id)
    return preds


if __name__ == '__main__':
    db_url = os.environ['APP_DB_URL']
    model_dir = os.environ.get('MODEL_DIR', '/models')
    target = date.today()
    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])
    predict_for_date(target, db_url, model_dir)
