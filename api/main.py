"""FastAPI application entry point for the Stud-Dash EDW platform."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database.pg_connection import init_db
from api.routers import auth, etl, datasets, kpi, predict, classify, cluster, forecast, reports, audit


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_title,
    description="Educational Data Warehouse and Business Intelligence Platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(etl.router)
app.include_router(datasets.router)
app.include_router(kpi.router)
app.include_router(predict.router)
app.include_router(classify.router)
app.include_router(cluster.router)
app.include_router(forecast.router)
app.include_router(reports.router)
app.include_router(audit.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
