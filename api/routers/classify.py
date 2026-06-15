"""Classification training router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from database.models import User
from database.pg_operations import (
    load_dataset_as_dataframe,
    list_model_runs,
    save_model_run,
    write_audit_log,
)
from models.classification import build_metrics_table, train_classification_models

router = APIRouter(prefix="/classify", tags=["classification"])


class TrainRequest(BaseModel):
    dataset_id: int
    feature_columns: list[str]
    target_column: str | None = None
    score_columns: list[str] | None = None
    pass_threshold: float = 60.0
    test_size: float = 0.3
    cv_folds: int = 5


class TrainResponse(BaseModel):
    run_id: int
    target_source: str
    train_rows: int
    test_rows: int
    performance_summary: str
    model_results: list[dict]


@router.post("/train", response_model=TrainResponse)
def train(
    body: TrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    data = load_dataset_as_dataframe(db, body.dataset_id)
    if data is None or data.empty:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    run = train_classification_models(
        data=data,
        feature_columns=body.feature_columns,
        target_column=body.target_column,
        score_columns=body.score_columns,
        pass_threshold=body.pass_threshold,
        test_size=body.test_size,
        cv_folds=body.cv_folds,
    )

    results_for_db = []
    for r in run.results:
        fi = r.feature_importance.to_dict(orient="records") if not r.feature_importance.empty else []
        cv = r.cross_validation_metrics.to_dict(orient="records") if not r.cross_validation_metrics.empty else []
        cm = r.confusion_matrix.to_dict()
        results_for_db.append({
            "model_name": r.model_name,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1_score": r.f1_score,
            "roc_auc": r.roc_auc,
            "average_precision": r.average_precision,
            "log_loss": r.log_loss,
            "brier_score": r.brier_score,
            "feature_importance": fi,
            "cv_metrics": cv,
            "confusion_matrix": cm,
            "probability_calibrated": r.probability_calibrated,
            "summary": r.summary,
        })

    db_run = save_model_run(
        db=db,
        dataset_id=body.dataset_id,
        run_type="classification",
        feature_columns=run.feature_columns,
        target_source=run.target_source,
        train_rows=run.train_rows,
        test_rows=run.test_rows,
        cv_folds=run.cv_folds,
        performance_summary=run.performance_summary,
        run_config={"pass_threshold": body.pass_threshold, "test_size": body.test_size},
        results=results_for_db,
    )

    write_audit_log(
        db, action="TRAIN_CLASSIFICATION", user_id=current_user.id,
        user_email=current_user.email, resource_type="model_run", resource_id=str(db_run.id),
    )

    return TrainResponse(
        run_id=db_run.id,
        target_source=run.target_source,
        train_rows=run.train_rows,
        test_rows=run.test_rows,
        performance_summary=run.performance_summary,
        model_results=results_for_db,
    )


@router.get("/runs", response_model=list[dict])
def list_runs(dataset_id: int | None = None, limit: int = 20, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    runs = list_model_runs(db, dataset_id=dataset_id, limit=limit)
    return [{"id": r.id, "dataset_id": r.dataset_id, "run_type": r.run_type, "target_source": r.target_source, "train_rows": r.train_rows, "test_rows": r.test_rows, "created_at": r.created_at} for r in runs]
