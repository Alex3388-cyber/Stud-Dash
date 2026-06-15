"""Service wrapper for forecasting."""

from __future__ import annotations

import pandas as pd

from models.forecasting import forecast_pass_rate, forecast_at_risk_trajectory


def generate_forecast(
    data: pd.DataFrame,
    forecast_type: str = "pass_rate",
    horizon_days: int = 90,
) -> dict:
    if forecast_type == "pass_rate":
        result = forecast_pass_rate(data, horizon_days=horizon_days)
    elif forecast_type == "at_risk":
        result = forecast_at_risk_trajectory(data, horizon_days=horizon_days)
    else:
        raise ValueError(f"Unknown forecast type: {forecast_type}. Use 'pass_rate' or 'at_risk'.")

    return {
        "method": result.method,
        "horizon_days": result.horizon_days,
        "data_points_used": result.data_points_used,
        "predictions": result.predictions,
        "model_params": result.model_params,
    }
