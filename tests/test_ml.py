"""
Tests for ml/ modules that do not require a database connection.
"""

import sys
import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _minimal_df(n=15):
    """Synthetic health_logs DataFrame with all required columns."""
    dates = pd.date_range("2024-01-01", periods=n).date
    return pd.DataFrame({
        "log_date":          dates,
        "weather":           ["sunny"] * n,
        "temperature":       [18] * n,
        "num_meals":         [3] * n,
        "num_junkfood":      [0] * n,
        "meal_size_lunch":   [3] * n,
        "meal_size_dinner":  [3] * n,
        "sugar":             [5] * n,
        "stomach_issues":    [1] * n,
        "sleep_quality":     [7] * n,
        "exercise_gym":      [False] * n,
        "exercise_morning":  [False] * n,
        "stress":            [3] * n,
        "social_activity":   [5] * n,
        "social_conflicts":  [1] * n,
        "tiredness_score":   [6] * n,
    })


# ── features.py tests ─────────────────────────────────────────────────────────

def test_feature_modules_importable():
    from ml.features import build_features, get_inference_features
    assert callable(build_features)
    assert callable(get_inference_features)


def test_build_features_returns_44_columns():
    from ml.features import build_features
    df = _minimal_df(15)
    X = build_features(df)
    assert X.shape == (15, 44), f"Expected (15, 44), got {X.shape}"


def test_build_features_all_float():
    from ml.features import build_features
    df = _minimal_df(5)
    X = build_features(df)
    assert X.dtypes.apply(lambda d: d == np.float64).all(), "All feature columns must be float64"


def test_build_features_no_nans():
    from ml.features import build_features
    df = _minimal_df(10)
    X = build_features(df)
    assert not X.isnull().any().any(), "build_features produced NaN values"


def test_build_features_single_row():
    """Rolling window with min_periods=1 must handle a single-row DataFrame."""
    from ml.features import build_features
    df = _minimal_df(1)
    X = build_features(df)
    assert X.shape == (1, 44)
    assert not X.isnull().any().any()


def test_build_features_weather_encoding():
    from ml.features import build_features, WEATHER_MAP
    df = _minimal_df(4)
    df["weather"] = ["sunny", "cloudy", "rainy", "snowy"]
    X = build_features(df)
    expected = [WEATHER_MAP[w] for w in ["sunny", "cloudy", "rainy", "snowy"]]
    assert list(X["weather_encoded"].astype(int)) == expected


def test_day_of_week_column():
    from ml.features import build_features
    df = _minimal_df(7)
    X = build_features(df)
    # 2024-01-01 is a Monday (0)
    assert int(X["day_of_week"].iloc[0]) == 0


def test_get_inference_features_excludes_target_date():
    """
    get_inference_features loads rows BEFORE target_date only.
    Verify by checking the SQL in features.py directly.
    """
    import ml.features as feat_module
    import inspect
    src = inspect.getsource(feat_module.get_inference_features)
    assert "log_date < :target" in src, "Must use < not <= to exclude target date"
    assert "log_date <= :target" not in src, "Must not include target date in feature window"


def test_get_inference_features_does_not_require_target_in_db():
    import ml.features as feat_module
    import inspect
    src = inspect.getsource(feat_module.get_inference_features)
    assert "target date itself not in DB" not in src
    assert "str(df['log_date'].max()) != str(target_date)" not in src


def test_tiredness_lag1_column_present():
    from ml.features import build_features
    df = _minimal_df(5)
    X = build_features(df)
    assert 'tiredness_lag1' in X.columns, "tiredness_lag1 feature must be present"


def test_tiredness_lag1_shifts_by_one_row():
    """Row i's tiredness_lag1 must equal row i-1's tiredness_score."""
    from ml.features import build_features
    df = _minimal_df(5)
    df['tiredness_score'] = [4, 6, 8, 5, 7]
    X = build_features(df)
    assert float(X['tiredness_lag1'].iloc[0]) == 5.0
    assert list(X['tiredness_lag1'].iloc[1:].astype(int)) == [4, 6, 8, 5]


def test_tiredness_lag1_no_nans_without_tiredness_score():
    """build_features must not produce NaN when tiredness_score column is absent."""
    from ml.features import build_features
    df = _minimal_df(5).drop(columns=['tiredness_score'])
    X = build_features(df)
    assert not X.isnull().any().any()
    assert (X['tiredness_lag1'] == 5.0).all(), "Should default to 5 when tiredness_score absent"


def test_get_inference_features_loads_tiredness_score():
    """get_inference_features SQL must include tiredness_score for lag-1 feature."""
    import ml.features as feat_module
    import inspect
    src = inspect.getsource(feat_module.get_inference_features)
    assert "tiredness_score" in src, "Must load tiredness_score for lag-1 feature"


# ── train.py tests ────────────────────────────────────────────────────────────

def test_train_module_importable():
    from ml.train import train_models
    assert callable(train_models)


def test_train_models_accepts_up_to_date_param():
    """train_models signature must accept up_to_date without error."""
    import inspect
    from ml.train import train_models
    params = inspect.signature(train_models).parameters
    assert 'up_to_date' in params


def test_build_models_from_params_returns_four_models(tmp_path):
    from ml.train import _build_models_from_params
    models = _build_models_from_params(str(tmp_path))
    assert set(models.keys()) == {'random_forest', 'gradient_boost', 'ridge', 'extra_trees'}


def test_build_models_from_params_uses_defaults_without_file(tmp_path):
    from ml.train import _build_models_from_params, _DEFAULT_PARAMS
    models = _build_models_from_params(str(tmp_path))
    assert models['random_forest'].n_estimators == _DEFAULT_PARAMS['random_forest']['n_estimators']
    assert models['ridge'].alpha == _DEFAULT_PARAMS['ridge']['alpha']


def test_build_models_from_params_merges_tuned_params(tmp_path):
    from ml.train import _build_models_from_params
    tuned = {'random_forest': {'n_estimators': 50}}
    (tmp_path / 'best_params.json').write_text(json.dumps(tuned))
    models = _build_models_from_params(str(tmp_path))
    assert models['random_forest'].n_estimators == 50


def test_build_models_from_params_does_not_affect_other_models(tmp_path):
    """Tuning one model must not change another model's params."""
    from ml.train import _build_models_from_params, _DEFAULT_PARAMS
    tuned = {'random_forest': {'n_estimators': 50}}
    (tmp_path / 'best_params.json').write_text(json.dumps(tuned))
    models = _build_models_from_params(str(tmp_path))
    assert models['extra_trees'].n_estimators == _DEFAULT_PARAMS['extra_trees']['n_estimators']


# ── predict.py tests ──────────────────────────────────────────────────────────

def test_predict_module_importable():
    from ml.predict import predict_for_date
    assert callable(predict_for_date)


# ── tune.py tests ─────────────────────────────────────────────────────────────

def test_tune_module_importable():
    from ml.tune import run_grid_search
    assert callable(run_grid_search)


def test_tune_load_best_scores_returns_empty_when_missing(tmp_path):
    from ml.tune import load_best_scores
    assert load_best_scores(str(tmp_path)) == {}


def test_tune_best_scores_roundtrip(tmp_path):
    from ml.tune import save_best_scores, load_best_scores
    scores = {'random_forest': 0.85, 'gradient_boost': 0.90}
    save_best_scores(str(tmp_path), scores)
    loaded = load_best_scores(str(tmp_path))
    assert loaded == scores


def test_tune_load_best_params_returns_empty_when_missing(tmp_path):
    from ml.tune import load_best_params
    assert load_best_params(str(tmp_path)) == {}


def test_tune_best_params_roundtrip(tmp_path):
    from ml.tune import save_best_params, load_best_params
    params = {'random_forest': {'n_estimators': 100, 'max_depth': 10}}
    save_best_params(str(tmp_path), params)
    loaded = load_best_params(str(tmp_path))
    assert loaded == params


def test_tune_best_params_null_max_depth_roundtrip(tmp_path):
    """max_depth=None must survive JSON serialisation (stored as null)."""
    from ml.tune import save_best_params, load_best_params
    params = {'random_forest': {'n_estimators': 100, 'max_depth': None}}
    save_best_params(str(tmp_path), params)
    loaded = load_best_params(str(tmp_path))
    assert loaded['random_forest']['max_depth'] is None


def test_tune_saved_params_apply_to_train(tmp_path):
    """Params written by tune must be picked up by _build_models_from_params."""
    from ml.tune import save_best_params
    from ml.train import _build_models_from_params
    save_best_params(str(tmp_path), {'ridge': {'alpha': 99.0}})
    models = _build_models_from_params(str(tmp_path))
    assert models['ridge'].alpha == 99.0
