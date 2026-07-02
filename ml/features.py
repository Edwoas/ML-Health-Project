"""
Feature engineering with a 10-day rolling window.

For each day t the feature vector contains:
  - The day's own raw values (14 features)
  - 10-day rolling mean of each value  (14 features)
  - 10-day rolling std  of each value  (14 features)
  - Day of week (0=Mon … 6=Sun)        (1 feature)

Total: 43 features per row.

Both build_features() (for training, full DataFrame) and
get_inference_features() (for a single target date, loads from DB)
share this logic.
"""

import pandas as pd
import numpy as np

WEATHER_MAP = {'sunny': 3, 'cloudy': 2, 'rainy': 1, 'snowy': 0}

# All numeric raw columns (weather handled separately)
RAW_NUMERIC = [
    'temperature',
    'num_meals',
    'num_junkfood',
    'meal_size_lunch',
    'meal_size_dinner',
    'sugar',
    'stomach_issues',
    'sleep_quality',
    'exercise_gym',
    'exercise_morning',
    'stress',
    'social_activity',
    'social_conflicts',
]

_DEFAULTS = {
    'temperature':      18,
    'num_meals':         3,
    'num_junkfood':      0,
    'meal_size_lunch':   3,
    'meal_size_dinner':  3,
    'sugar':             5,
    'stomach_issues':    1,
    'sleep_quality':     7,
    'exercise_gym':      0,
    'exercise_morning':  0,
    'stress':            3,
    'social_activity':   5,
    'social_conflicts':  1,
}

LAG_WINDOW = 10


def _prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and type-cast a copy of df; sort by log_date."""
    df = df.copy().sort_values('log_date').reset_index(drop=True)

    df['weather_encoded'] = (
        df['weather'].astype(str).str.lower().str.strip()
        .map(WEATHER_MAP).fillna(2).astype(float)
    )

    for col in ('exercise_gym', 'exercise_morning'):
        df[col] = df[col].fillna(False).astype(int).astype(float)

    for col in RAW_NUMERIC:
        if col not in ('exercise_gym', 'exercise_morning'):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(_DEFAULTS.get(col, 0))

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix for an entire DataFrame.
    Input must contain all raw columns plus log_date.
    Returns a DataFrame with 43 columns, same index as the sorted df.
    Caller should drop rows with NaN targets before fitting.
    """
    df = _prepare_raw(df)

    all_raw = ['weather_encoded'] + RAW_NUMERIC
    result = {}

    # 1. Current-day raw values
    for col in all_raw:
        result[col] = df[col].values

    # 2. Rolling mean and std over LAG_WINDOW days (min_periods=1 avoids NaN on early rows)
    for col in all_raw:
        s = df[col]
        result[f'{col}_mean{LAG_WINDOW}'] = (
            s.rolling(LAG_WINDOW, min_periods=1).mean().values
        )
        result[f'{col}_std{LAG_WINDOW}'] = (
            s.rolling(LAG_WINDOW, min_periods=1).std().fillna(0).values
        )

    # 3. Day of week (0 = Monday)
    result['day_of_week'] = pd.to_datetime(df['log_date']).dt.dayofweek.values

    return pd.DataFrame(result).astype(float)


def get_inference_features(target_date, db_url: str) -> 'pd.DataFrame | None':
    """
    Load the last LAG_WINDOW scored days BEFORE target_date from the DB and
    return a feature row representing the historical context for predicting
    target_date. target_date itself is never loaded — its data is unseen.
    Returns None if there are no scored rows before target_date.
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text(
                "SELECT log_date, weather, temperature, num_meals, num_junkfood, "
                "meal_size_lunch, meal_size_dinner, sugar, "
                "stomach_issues, sleep_quality, exercise_gym, exercise_morning, "
                "stress, social_activity, social_conflicts "
                "FROM health_logs "
                "WHERE log_date < :target "
                f"ORDER BY log_date DESC LIMIT {LAG_WINDOW}"
            ),
            {'target': str(target_date)},
        )
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df.empty:
        return None

    df = df.sort_values('log_date').reset_index(drop=True)
    features = build_features(df)
    return features.iloc[[-1]]
