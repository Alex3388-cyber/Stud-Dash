from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    study_time: float = Field(..., ge=0, description="Weekly study hours")
    absences: float = Field(..., ge=0, description="Number of absences or attendance proxy")
    failures: float = Field(0, ge=0, description="Number of past failures")
    previous_grade_1: float = Field(..., description="First previous grade")
    previous_grade_2: float = Field(..., description="Second previous grade")
    dataset_id: int | None = None
    include_shap: bool = False


class PredictionResponse(BaseModel):
    predicted_label: str
    pass_probability: float
    fail_probability: float
    confidence_score: float
    risk_level: str
    recommendation: str
    shap_values: dict | None = None
    prediction_id: int | None = None


class BatchPredictionRequest(BaseModel):
    records: list[dict]
    dataset_id: int | None = None


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
    pass_count: int
    fail_count: int


class PredictionEventOut(BaseModel):
    id: int
    model_name: str
    input_features: dict
    predicted_label: str
    pass_probability: float
    fail_probability: float
    risk_level: str
    recommendation: str
    created_at: datetime

    model_config = {"from_attributes": True}
