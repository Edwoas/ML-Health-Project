from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter(prefix='/prediction-history', tags=['prediction-history'])


class HistoryEntry(BaseModel):
    run_at:              datetime
    pred_random_forest:  Optional[float] = None
    pred_gradient_boost: Optional[float] = None
    pred_ridge:          Optional[float] = None
    pred_extra_trees:    Optional[float] = None
    pred_ensemble:       float
    model_version:       Optional[str]   = None


@router.get('/dates', response_model=List[str])
def get_dates_with_history(db: Session = Depends(get_db)):
    """All log_dates that have at least one history entry, newest first."""
    rows = db.execute(text("""
        SELECT DISTINCT log_date
        FROM prediction_history
        ORDER BY log_date DESC
    """)).fetchall()
    return [str(r[0]) for r in rows]


@router.get('/', response_model=List[HistoryEntry])
def get_history_for_date(
    log_date: date = Query(..., description='Date to inspect (YYYY-MM-DD)'),
    db: Session = Depends(get_db),
):
    """All inference runs for a given log_date, oldest first — use to plot model convergence."""
    rows = db.execute(text("""
        SELECT run_at,
               pred_random_forest, pred_gradient_boost,
               pred_ridge, pred_extra_trees,
               pred_ensemble, model_version
        FROM prediction_history
        WHERE log_date = :log_date
        ORDER BY run_at ASC
    """), {'log_date': str(log_date)}).fetchall()

    return [HistoryEntry(
        run_at=row[0],
        pred_random_forest=row[1],
        pred_gradient_boost=row[2],
        pred_ridge=row[3],
        pred_extra_trees=row[4],
        pred_ensemble=row[5],
        model_version=row[6],
    ) for row in rows]
