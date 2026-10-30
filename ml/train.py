"""
Train four regressors (Random Forest, Gradient Boosting, Ridge, Extra Trees)
on all health_logs rows that have a tiredness_score, using 10-day rolling
lag features.  All four models are saved together as a single dict in
MODEL_DIR/models.pkl (plus a timestamped archive copy).

Hyperparameters are loaded from MODEL_DIR/best_params.json if it exists
(written by ml/tune.py after a grid search run); otherwise defaults are used.
"""

import json
import os
import sys
import joblib
from datetime import datetime

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sqlalchemy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_SAMPLES = 10

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_EXPERIMENT   = 'health_tiredness_prediction'

_ESTIMATOR_CLASSES = {
    'random_forest':  RandomForestRegressor,
    'gradient_boost': GradientBoostingRegressor,
    'ridge':          Ridge,
    'extra_trees':    ExtraTreesRegressor,
}

_DEFAULT_PARAMS = {
    'random_forest':  {'n_estimators': 200, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1},
    'gradient_boost': {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'random_state': 42},
    'ridge':          {'alpha': 1.0},
    'extra_trees':    {'n_estimators': 200, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1},
}


def _make_pipeline(estimator):
    return Pipeline([('scaler', StandardScaler()), ('model', estimator)])


def _build_models_from_params(model_dir: str) -> dict:
    """Build model instances, merging tuned params from best_params.json if present."""
    best_params_path = os.path.join(model_dir, 'best_params.json')
    tuned = {}
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            tuned = json.load(f)
    return {
        name: cls(**{**_DEFAULT_PARAMS[name], **tuned.get(name, {})})
        for name, cls in _ESTIMATOR_CLASSES.items()
    }


def _log_training_to_mlflow(up_to_date, n_samples, cv_results: dict,
                             version: str, min_date=None, max_date=None):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"train_{up_to_date}") as run:
            mlflow.log_param('up_to_date',    str(up_to_date))
            mlflow.log_param('n_samples',     n_samples)
            mlflow.log_param('model_version', version)
            if min_date is not None and max_date is not None:
                try:
                    mlflow.log_param('training_date_range_days', (max_date - min_date).days)
                except Exception:
                    pass

            maes = []
            for name, m in cv_results.items():
                mlflow.log_metric(f'{name}_cv_mae',     m['mae'])
                mlflow.log_metric(f'{name}_cv_mae_std', m['mae_std'])
                mlflow.log_metric(f'{name}_cv_r2',      m['r2'])
                maes.append(m['mae'])

            if maes:
                mlflow.log_metric('ensemble_cv_mae_approx', sum(maes) / len(maes))

            return run.info.run_id
    except Exception as e:
        print(f"MLflow training log failed (non-fatal): {e}")
        return None


def train_models(db_url: str, model_dir: str = '/models', up_to_date=None) -> dict:
    """
    Train on all scored rows up to and including up_to_date.
    If up_to_date is None, uses all available scored rows.
    """
    from ml.features import build_features

    engine = sqlalchemy.create_engine(db_url)
    query = """
        SELECT log_date, weather, temperature, num_meals, num_junkfood,
               meal_size_lunch, meal_size_dinner, sugar,
               stomach_issues, sleep_quality, exercise_gym, exercise_morning,
               stress, social_activity, social_conflicts, tiredness_score
        FROM health_logs
        WHERE tiredness_score IS NOT NULL
        {date_filter}
        ORDER BY log_date
    """
    if up_to_date is not None:
        sql    = sqlalchemy.text(query.format(date_filter="AND log_date <= :up_to_date"))
        params = {'up_to_date': str(up_to_date)}
    else:
        sql    = sqlalchemy.text(query.format(date_filter=""))
        params = {}

    with engine.connect() as conn:
        result = conn.execute(sql, params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if len(df) < MIN_SAMPLES:
        print(f"Only {len(df)} labelled rows (need >= {MIN_SAMPLES}). Skipping training.")
        return {'status': 'skipped', 'reason': 'insufficient_data', 'count': len(df)}

    X = build_features(df)
    y = df['tiredness_score'].astype(float).values

    cv_folds  = min(5, len(df) // 3)
    cv_results = {}
    trained    = {}

    for name, estimator in _build_models_from_params(model_dir).items():
        pipeline = _make_pipeline(estimator)

        if cv_folds >= 2:
            cv_out  = cross_validate(
                pipeline, X, y, cv=cv_folds,
                scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'},
            )
            mae     = float(-cv_out['test_mae'].mean())
            mae_std = float(cv_out['test_mae'].std())
            r2      = float(cv_out['test_r2'].mean())
            cv_results[name] = {'mae': mae, 'mae_std': mae_std, 'r2': r2}
            print(f"[{name}] CV MAE: {mae:.3f} ± {mae_std:.3f}  R²: {r2:.3f}")

        pipeline.fit(X, y)
        trained[name] = pipeline
        print(f"[{name}] trained on {len(df)} samples.")

    os.makedirs(model_dir, exist_ok=True)
    version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    joblib.dump(trained, os.path.join(model_dir, 'models.pkl'))
    joblib.dump(trained, os.path.join(model_dir, f'models_{version}.pkl'))
    print(f"All models saved. Version: {version}")

    min_date = df['log_date'].min()
    max_date = df['log_date'].max()
    try:
        if hasattr(min_date, 'date'):
            min_date = min_date.date()
        if hasattr(max_date, 'date'):
            max_date = max_date.date()
    except Exception:
        min_date = max_date = None

    mlflow_run_id = _log_training_to_mlflow(
        up_to_date, len(df), cv_results, version, min_date, max_date
    )

    return {
        'status':        'success',
        'version':       version,
        'n_samples':     len(df),
        'mlflow_run_id': mlflow_run_id,
    }


if __name__ == '__main__':
    db_url    = os.environ['APP_DB_URL']
    model_dir = os.environ.get('MODEL_DIR', '/models')
    train_models(db_url, model_dir)
