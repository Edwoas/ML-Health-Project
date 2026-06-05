from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel


class HealthLogOut(BaseModel):
    id:                 int
    log_date:           date
    weather:            Optional[str]  = None
    temperature:        Optional[int]  = None
    num_meals:          Optional[int]  = None
    num_junkfood:       Optional[int]  = None
    meal_size_lunch:    Optional[int]  = None
    meal_size_dinner:   Optional[int]  = None
    sugar:              Optional[int]  = None
    stomach_issues:     Optional[int]  = None
    sleep_quality:      Optional[int]  = None
    exercise_gym:       Optional[bool] = None
    exercise_morning:   Optional[bool] = None
    stress:             Optional[int]  = None
    social_activity:    Optional[int]  = None
    social_conflicts:   Optional[int]  = None
    tiredness_score:    Optional[int]  = None
    created_at:         Optional[datetime] = None
    updated_at:         Optional[datetime] = None

    model_config = {'from_attributes': True}


class DashboardEntry(BaseModel):
    log_date:            date
    weather:             Optional[str]   = None
    temperature:         Optional[int]   = None
    num_meals:           Optional[int]   = None
    num_junkfood:        Optional[int]   = None
    meal_size_lunch:     Optional[int]   = None
    meal_size_dinner:    Optional[int]   = None
    sugar:               Optional[int]   = None
    stomach_issues:      Optional[int]   = None
    sleep_quality:       Optional[int]   = None
    exercise_gym:        Optional[bool]  = None
    exercise_morning:    Optional[bool]  = None
    stress:              Optional[int]   = None
    social_activity:     Optional[int]   = None
    social_conflicts:    Optional[int]   = None
    tiredness_score:     Optional[int]   = None
    pred_random_forest:  Optional[float] = None
    pred_gradient_boost: Optional[float] = None
    pred_ridge:          Optional[float] = None
    pred_extra_trees:    Optional[float] = None
    pred_ensemble:       Optional[float] = None
    model_version:       Optional[str]   = None
