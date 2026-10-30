"""
Grid search hyperparameter tuning: champion/challenger comparison.

Reads current best params from MODEL_DIR/best_params.json (defaults used if absent).
Writes updated params + scores if any model improves.
Writes MODEL_DIR/last_tuned.txt so the DAG can gate whether to re-run.
Logs results to MLflow.
"""

import json
import os
import time
from datetime import date

import pandas as pd
import sqlalchemy
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_EXPERIMENT   = 'health_tiredness_prediction'

PARAM_GRIDS = {
    'random_forest': {
        'model__n_estimators': [100, 200],
        'model__max_depth':    [None, 10],
        'model__max_features': ['sqrt', 'log2'],
    },
    'gradient_boost': {
        'model__n_estimators':  [100, 150],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth':     [2, 3],
    },
    'ridge': {
        'model__alpha': [0.1, 1.0, 10.0],
    },
    'extra_trees': {
        'model__n_estimators': [100, 200],
        'model__max_depth':    [None, 10],
        'model__max_features': ['sqrt', 'log2'],
    },
}

_BASE_ESTIMATORS = {
    'random_forest':  RandomForestRegressor(random_state=42, n_jobs=1),
    'gradient_boost': GradientBoostingRegressor(random_state=42),
    'ridge':          Ridge(),
    'extra_trees':    ExtraTreesRegressor(random_state=42, n_jobs=1),
}


def _make_pipeline(estimator):
    return Pipeline([('scaler', StandardScaler()), ('model', estimator)])


def load_best_scores(model_dir: str) -> dict:
    path = os.path.join(model_dir, 'best_scores.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_best_scores(model_dir: str, scores: dict):
    with open(os.path.join(model_dir, 'best_scores.json'), 'w') as f:
        json.dump(scores, f, indent=2)


def load_best_params(model_dir: str) -> dict:
    path = os.path.join(model_dir, 'best_params.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_best_params(model_dir: str, params: dict):
    with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
        json.dump(params, f, indent=2)


def run_grid_search(db_url: str, model_dir: str = '/models') -> dict:
    from ml.features import build_features

    engine = sqlalchemy.create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("""
            SELECT log_date, weather, temperature, num_meals, num_junkfood,
                   meal_size_lunch, meal_size_dinner, sugar,
                   stomach_issues, sleep_quality, exercise_gym, exercise_morning,
                   stress, social_activity, social_conflicts, tiredness_score
            FROM health_logs
            WHERE tiredness_score IS NOT NULL
            ORDER BY log_date
        """))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if len(df) < 10:
        print(f"Only {len(df)} labelled rows — skipping grid search.")
        return {'status': 'skipped', 'reason': 'insufficient_data', 'count': len(df)}

    X = build_features(df)
    y = df['tiredness_score'].astype(float).values

    cv_folds = min(3, len(df) // 5)
    if cv_folds < 2:
        return {'status': 'skipped', 'reason': 'too_few_for_cv', 'count': len(df)}

    current_best   = load_best_scores(model_dir)
    current_params = load_best_params(model_dir)

    new_best_scores = dict(current_best)
    new_best_params = dict(current_params)
    improved_models = []
    gs_results      = {}
    total_fits      = 0
    start_time      = time.time()

    for name, grid in PARAM_GRIDS.items():
        pipeline = _make_pipeline(_BASE_ESTIMATORS[name])
        gs = GridSearchCV(
            pipeline, grid, cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=1, refit=False,
        )
        gs.fit(X, y)
        best_mae   = float(-gs.best_score_)
        best_params = gs.best_params_
        total_fits += len(gs.cv_results_['mean_test_score']) * cv_folds

        previous_mae = current_best.get(name)
        improved     = previous_mae is None or best_mae < previous_mae

        gs_results[name] = {
            'best_mae':    best_mae,
            'best_params': best_params,
            'previous_mae': previous_mae,
            'improved':    improved,
            'delta':       None if previous_mae is None else previous_mae - best_mae,
        }

        if improved:
            new_best_scores[name] = best_mae
            clean = {k.replace('model__', ''): v for k, v in best_params.items()}
            new_best_params[name] = clean
            improved_models.append(name)

        prev_str = f"{previous_mae:.3f}" if previous_mae is not None else "N/A"
        print(
            f"[{name}] best_mae={best_mae:.3f}  prev={prev_str}"
            f"  improved={improved}  params={best_params}"
        )

    elapsed = time.time() - start_time
    os.makedirs(model_dir, exist_ok=True)

    if improved_models:
        save_best_params(model_dir, new_best_params)
        save_best_scores(model_dir, new_best_scores)
        print(f"Champion updated for: {improved_models}")
    else:
        print("No improvement found — params unchanged.")

    with open(os.path.join(model_dir, 'last_tuned.txt'), 'w') as f:
        f.write(str(date.today()))

    _log_tuning_to_mlflow(gs_results, improved_models, total_fits, elapsed, len(df))

    return {
        'status':          'success',
        'any_improved':    bool(improved_models),
        'improved_models': improved_models,
        'total_fits':      total_fits,
        'elapsed_seconds': round(elapsed, 1),
    }


def _log_tuning_to_mlflow(results: dict, improved_models: list,
                           total_fits: int, elapsed: float, n_samples: int):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"grid_search_{date.today()}"):
            mlflow.log_param('n_samples',       n_samples)
            mlflow.log_param('total_fits',      total_fits)
            mlflow.log_param('improved_models', ','.join(improved_models) or 'none')
            mlflow.log_metric('elapsed_seconds', round(elapsed, 1))
            for name, r in results.items():
                mlflow.log_metric(f'{name}_best_mae', r['best_mae'])
                if r['delta'] is not None:
                    mlflow.log_metric(f'{name}_mae_delta', r['delta'])
                for k, v in r['best_params'].items():
                    clean_k = k.replace('model__', '')
                    try:
                        mlflow.log_param(f'{name}_{clean_k}', v)
                    except Exception:
                        pass
    except Exception as e:
        print(f"MLflow grid search log failed (non-fatal): {e}")
