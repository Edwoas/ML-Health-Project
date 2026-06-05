"""
Compute Pearson correlations between each health metric and tiredness_score,
then return them with anonymised labels so the chart doesn't bias future logging.

Label assignment uses a fixed seed so the same metric always gets the same letter
across page loads, but the mapping is opaque to the viewer.
"""

import math
import random
from typing import List

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter(prefix='/correlations', tags=['correlations'])

MIN_ROWS = 5  # refuse to correlate on fewer than this

# All numeric health metrics (booleans treated as 0/1)
_METRICS = [
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

# Build a stable, shuffled label map with a fixed seed.
# The seed is arbitrary — the goal is just that the mapping never changes.
def _build_label_map() -> dict[str, str]:
    rng = random.Random(7492816)
    order = _METRICS[:]
    rng.shuffle(order)
    letters = [chr(65 + i) for i in range(len(order))]   # A, B, C, …
    return {metric: letter for metric, letter in zip(order, letters)}

_LABEL = _build_label_map()


class CorrelationItem(BaseModel):
    label: str
    correlation: float


def _pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson r, dropping NaN pairs. Returns None when sample is too small."""
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 3:
        return None
    xm = xa - xa.mean()
    ym = ya - ya.mean()
    denom = math.sqrt(float((xm ** 2).sum()) * float((ym ** 2).sum()))
    return float((xm * ym).sum() / denom) if denom != 0 else 0.0


@router.get('/', response_model=List[CorrelationItem])
def get_correlations(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT
            temperature,
            num_meals,
            num_junkfood,
            meal_size_lunch,
            meal_size_dinner,
            sugar,
            stomach_issues,
            sleep_quality,
            exercise_gym::int,
            exercise_morning::int,
            stress,
            social_activity,
            social_conflicts,
            tiredness_score
        FROM health_logs
        WHERE tiredness_score IS NOT NULL
    """)).fetchall()

    if len(rows) < MIN_ROWS:
        return []

    # Transpose rows → per-column lists
    cols = _METRICS + ['tiredness_score']
    data: dict[str, list] = {c: [] for c in cols}
    for row in rows:
        for i, col in enumerate(cols):
            v = row[i]
            data[col].append(float(v) if v is not None else float('nan'))

    tiredness = data['tiredness_score']
    results: list[CorrelationItem] = []

    for metric in _METRICS:
        r = _pearson(data[metric], tiredness)
        if r is None:
            continue
        results.append(CorrelationItem(
            label=_LABEL[metric],
            correlation=round(r, 3),
        ))

    # Sort by absolute correlation descending so the most informative bars are on top
    results.sort(key=lambda item: abs(item.correlation), reverse=True)
    return results
