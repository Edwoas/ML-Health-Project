from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import HealthLog
from ..schemas import HealthLogOut

router = APIRouter(prefix='/logs', tags=['logs'])


@router.get('/', response_model=List[HealthLogOut])
def list_logs(
    db: Session = Depends(get_db),
    limit: int = Query(100, le=365),
    offset: int = 0,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    q = db.query(HealthLog).order_by(HealthLog.log_date.desc())
    if start_date:
        q = q.filter(HealthLog.log_date >= start_date)
    if end_date:
        q = q.filter(HealthLog.log_date <= end_date)
    return q.offset(offset).limit(limit).all()
