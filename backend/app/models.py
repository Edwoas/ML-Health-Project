from sqlalchemy import Boolean, Column, Date, DateTime, Float, Integer, String, func
from .database import Base


class HealthLog(Base):
    __tablename__ = 'health_logs'

    id                 = Column(Integer, primary_key=True)
    log_date           = Column(Date, unique=True, nullable=False)
    weather            = Column(String(20))
    temperature        = Column(Integer)
    num_meals          = Column(Integer)
    num_junkfood       = Column(Integer)
    meal_size_lunch    = Column(Integer)
    meal_size_dinner   = Column(Integer)
    sugar              = Column(Integer)
    stomach_issues     = Column(Integer)
    sleep_quality      = Column(Integer)
    exercise_gym       = Column(Boolean)
    exercise_morning   = Column(Boolean)
    stress             = Column(Integer)
    social_activity    = Column(Integer)
    social_conflicts   = Column(Integer)
    tiredness_score    = Column(Integer)
    created_at         = Column(DateTime, server_default=func.now())
    updated_at         = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Prediction(Base):
    __tablename__ = 'predictions'

    id                  = Column(Integer, primary_key=True)
    log_date            = Column(Date, unique=True, nullable=False)
    pred_random_forest  = Column(Float)
    pred_gradient_boost = Column(Float)
    pred_ridge          = Column(Float)
    pred_extra_trees    = Column(Float)
    pred_ensemble       = Column(Float, nullable=False)
    model_version       = Column(String(50))
    created_at          = Column(DateTime, server_default=func.now())


class PredictionHistory(Base):
    __tablename__ = 'prediction_history'

    id                  = Column(Integer, primary_key=True)
    log_date            = Column(Date, nullable=False, index=True)
    pred_random_forest  = Column(Float)
    pred_gradient_boost = Column(Float)
    pred_ridge          = Column(Float)
    pred_extra_trees    = Column(Float)
    pred_ensemble       = Column(Float, nullable=False)
    model_version       = Column(String(50))
    run_at              = Column(DateTime, server_default=func.now())
