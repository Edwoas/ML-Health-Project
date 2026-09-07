from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import HealthLog, Prediction
from ..schemas import DashboardEntry

router = APIRouter(prefix='/dashboard', tags=['dashboard'])

_KEYS = [
    'log_date', 'weather', 'temperature', 'num_meals', 'num_junkfood',
    'meal_size_lunch', 'meal_size_dinner', 'sugar',
    'stomach_issues', 'sleep_quality', 'exercise_gym', 'exercise_morning',
    'stress', 'social_activity', 'social_conflicts', 'tiredness_score',
    'pred_random_forest', 'pred_gradient_boost', 'pred_ridge',
    'pred_extra_trees', 'pred_ensemble', 'model_version',
]


@router.get('/', response_model=List[DashboardEntry])
def get_dashboard(
    db: Session = Depends(get_db),
    limit: int = Query(500, le=3650),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    q = (
        db.query(
            HealthLog.log_date,
            HealthLog.weather,
            HealthLog.temperature,
            HealthLog.num_meals,
            HealthLog.num_junkfood,
            HealthLog.meal_size_lunch,
            HealthLog.meal_size_dinner,
            HealthLog.sugar,
            HealthLog.stomach_issues,
            HealthLog.sleep_quality,
            HealthLog.exercise_gym,
            HealthLog.exercise_morning,
            HealthLog.stress,
            HealthLog.social_activity,
            HealthLog.social_conflicts,
            HealthLog.tiredness_score,
            Prediction.pred_random_forest,
            Prediction.pred_gradient_boost,
            Prediction.pred_ridge,
            Prediction.pred_extra_trees,
            Prediction.pred_ensemble,
            Prediction.model_version,
        )
        .outerjoin(Prediction, HealthLog.log_date == Prediction.log_date)
        .order_by(HealthLog.log_date.desc())
    )

    if start_date:
        q = q.filter(HealthLog.log_date >= start_date)
    if end_date:
        q = q.filter(HealthLog.log_date <= end_date)

    return [DashboardEntry(**dict(zip(_KEYS, row))) for row in q.limit(limit).all()]
