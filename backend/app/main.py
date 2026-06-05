from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .routes.correlations import router as correlations_router
from .routes.history import router as history_router
from .routes.logs import router as logs_router
from .routes.predictions import router as dashboard_router

# Create tables if they don't exist yet (idempotent)
Base.metadata.create_all(bind=engine)

app = FastAPI(title='Health ML API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(logs_router)
app.include_router(dashboard_router)
app.include_router(correlations_router)
app.include_router(history_router)


@app.get('/health')
def health_check():
    return {'status': 'ok'}
