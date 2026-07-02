"""
Tests for ml/ modules that do not require a database connection.
"""

import sys
import os
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


def test_feature_modules_importable():
    from ml.features import build_features, get_inference_features
    assert callable(build_features)
    assert callable(get_inference_features)


def test_train_module_importable():
    from ml.train import train_models
    assert callable(train_models)


def test_predict_module_importable():
    from ml.predict import predict_for_date
    assert callable(predict_for_date)


def test_build_features_returns_43_columns():
    from ml.features import build_features
    df = _minimal_df(15)
    X = build_features(df)
    assert X.shape == (15, 43), f"Expected (15, 43), got {X.shape}"


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
    assert X.shape == (1, 43)
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


def test_train_models_accepts_up_to_date_param():
    """train_models signature must accept up_to_date without error."""
    import inspect
    from ml.train import train_models
    params = inspect.signature(train_models).parameters
    assert 'up_to_date' in params


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
    """
    get_inference_features must not check whether target_date itself
    exists in health_logs.
    """
    import ml.features as feat_module
    import inspect
    src = inspect.getsource(feat_module.get_inference_features)
    assert "target date itself not in DB" not in src
    assert "str(df['log_date'].max()) != str(target_date)" not in src
