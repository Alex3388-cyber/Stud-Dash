"""Forecasting router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from database.models import User
from database.pg_operations import get_dataset_upload, get_latest_forecast, load_dataset_as_dataframe, save_forecast
from services.forecasting_service import generate_forecast

router = APIRouter(prefix="/forecast", tags=["forecasting"])


class ForecastRequest(BaseModel):
    dataset_id: int
    forecast_type: str = "pass_rate"
    horizon_days: int = 90


class ForecastResponse(BaseModel):
    dataset_id: int
    forecast_type: str
    method: str
    horizon_days: int
    data_points_used: int
    predictions: dict


@router.post("/generate", response_model=ForecastResponse)
def generate(
    body: ForecastRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    data = load_dataset_as_dataframe(db, body.dataset_id)
    if data is None or data.empty:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    result = generate_forecast(data, forecast_type=body.forecast_type, horizon_days=body.horizon_days)

    save_forecast(
        db=db,
        dataset_id=body.dataset_id,
        forecast_type=body.forecast_type,
        method=result["method"],
        horizon_days=body.horizon_days,
        data_points_used=result["data_points_used"],
        predictions=result["predictions"],
        model_params=result.get("model_params"),
    )

    return ForecastResponse(
        dataset_id=body.dataset_id,
        forecast_type=body.forecast_type,
        method=result["method"],
        horizon_days=body.horizon_days,
        data_points_used=result["data_points_used"],
        predictions=result["predictions"],
    )


@router.get("/{dataset_id}/{forecast_type}", response_model=ForecastResponse)
def get_forecast(
    dataset_id: int,
    forecast_type: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    row = get_latest_forecast(db, dataset_id, forecast_type)
    if not row:
        raise HTTPException(status_code=404, detail="No forecast found. POST /forecast/generate first.")
    return ForecastResponse(
        dataset_id=row.dataset_id,
        forecast_type=row.forecast_type,
        method=row.method,
        horizon_days=row.horizon_days,
        data_points_used=row.data_points_used,
        predictions=row.predictions,
    )
