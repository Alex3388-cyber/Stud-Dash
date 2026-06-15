"""Prediction router — single and batch student predictions."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user, get_optional_user
from api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionEventOut,
)
from database.models import User
from database.pg_operations import (
    get_dataset_upload,
    list_prediction_events,
    load_dataset_as_dataframe,
    save_prediction_event,
    write_audit_log,
)
from models.student_prediction import train_logistic_prediction_model, predict_student_performance

router = APIRouter(prefix="/predict", tags=["predictions"])

_model_cache: dict[int, object] = {}


def _get_model(db: Session, dataset_id: int):
    if dataset_id not in _model_cache:
        data = load_dataset_as_dataframe(db, dataset_id)
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="Dataset not found or empty")
        _model_cache[dataset_id] = train_logistic_prediction_model(data)
    return _model_cache[dataset_id]


@router.post("/single", response_model=PredictionResponse)
def predict_single(
    body: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    t0 = time.time()

    if body.dataset_id:
        try:
            bundle = _get_model(db, body.dataset_id)
        except HTTPException:
            raise
    else:
        latest = db.query(__import__("database.models", fromlist=["DatasetUpload"]).DatasetUpload)\
            .order_by(__import__("database.models", fromlist=["DatasetUpload"]).DatasetUpload.created_at.desc())\
            .first()
        if not latest:
            raise HTTPException(status_code=400, detail="No dataset available. Upload a dataset first.")
        bundle = _get_model(db, latest.id)
        body = body.model_copy(update={"dataset_id": latest.id})

    result = predict_student_performance(
        bundle,
        study_time=body.study_time,
        absences=body.absences,
        failures=body.failures,
        previous_grade_1=body.previous_grade_1,
        previous_grade_2=body.previous_grade_2,
    )

    shap_values = None
    if body.include_shap:
        try:
            from models.explainability import explain_prediction
            import pandas as pd
            features = pd.DataFrame([{
                "study_time": body.study_time,
                "absences": body.absences,
                "failures": body.failures,
                "previous_grade_1": body.previous_grade_1,
                "previous_grade_2": body.previous_grade_2,
            }])
            shap_values = explain_prediction(bundle.model, features)
        except Exception:
            pass

    event = save_prediction_event(
        db=db,
        model_name="Logistic Regression",
        input_features={
            "study_time": body.study_time,
            "absences": body.absences,
            "failures": body.failures,
            "previous_grade_1": body.previous_grade_1,
            "previous_grade_2": body.previous_grade_2,
        },
        predicted_label=result.predicted_label,
        pass_probability=result.pass_probability,
        fail_probability=result.fail_probability,
        confidence_score=result.confidence_score,
        risk_level=result.risk_level,
        recommendation=result.recommendation,
        user_id=current_user.id if current_user else None,
        dataset_id=body.dataset_id,
        shap_values=shap_values,
    )

    duration_ms = int((time.time() - t0) * 1000)
    write_audit_log(
        db, action="PREDICT", user_id=current_user.id if current_user else None,
        resource_type="prediction", resource_id=str(event.id), response_status=200, duration_ms=duration_ms,
    )

    return PredictionResponse(
        predicted_label=result.predicted_label,
        pass_probability=result.pass_probability,
        fail_probability=result.fail_probability,
        confidence_score=result.confidence_score,
        risk_level=result.risk_level,
        recommendation=result.recommendation,
        shap_values=shap_values,
        prediction_id=event.id,
    )


@router.get("/history", response_model=list[PredictionEventOut])
def prediction_history(
    limit: int = 100,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return list_prediction_events(db, limit=limit)
