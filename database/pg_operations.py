"""High-level database operations for all EDW entities."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from database.models import (
    AuditLog,
    DatasetUpload,
    EtlJob,
    ForecastResult,
    KpiSnapshot,
    ModelResult,
    ModelRun,
    PredictionEvent,
    Report,
    Session as DbSession,
    StudentRecord,
    User,
    UserRole,
)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(db: Session, email: str, hashed_password: str, full_name: str | None = None, role: UserRole = UserRole.viewer) -> User:
    user = User(email=email, hashed_password=hashed_password, full_name=full_name, role=role)
    db.add(user)
    db.flush()
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def update_last_login(db: Session, user_id: int) -> None:
    db.query(User).filter(User.id == user_id).update({"last_login": datetime.now(timezone.utc)})


def list_users(db: Session, limit: int = 100) -> list[User]:
    return db.query(User).order_by(User.created_at.desc()).limit(limit).all()


# ---------------------------------------------------------------------------
# Sessions / tokens
# ---------------------------------------------------------------------------

def create_session(db: Session, user_id: int, token: str, refresh_token: str, expires_at: datetime) -> DbSession:
    s = DbSession(user_id=user_id, token=token, refresh_token=refresh_token, expires_at=expires_at)
    db.add(s)
    db.flush()
    return s


def get_session_by_token(db: Session, token: str) -> DbSession | None:
    return db.query(DbSession).filter(DbSession.token == token, DbSession.revoked == False).first()  # noqa: E712


def revoke_session(db: Session, token: str) -> None:
    db.query(DbSession).filter(DbSession.token == token).update({"revoked": True})


def revoke_all_user_sessions(db: Session, user_id: int) -> None:
    db.query(DbSession).filter(DbSession.user_id == user_id).update({"revoked": True})


# ---------------------------------------------------------------------------
# Dataset uploads
# ---------------------------------------------------------------------------

def save_dataset_upload(
    db: Session,
    file_name: str,
    data: pd.DataFrame,
    uploaded_by: int | None = None,
    file_size_bytes: int | None = None,
) -> DatasetUpload:
    upload = DatasetUpload(
        file_name=file_name,
        uploaded_by=uploaded_by,
        row_count=len(data),
        column_count=len(data.columns),
        column_names=list(data.columns),
        file_size_bytes=file_size_bytes,
        status="loaded",
    )
    db.add(upload)
    db.flush()

    records = [
        StudentRecord(
            dataset_id=upload.id,
            row_number=i + 1,
            data={k: (None if pd.isna(v) else v) for k, v in row.items()},
        )
        for i, row in enumerate(data.to_dict(orient="records"))
    ]
    db.bulk_save_objects(records)
    return upload


def get_dataset_upload(db: Session, dataset_id: int) -> DatasetUpload | None:
    return db.query(DatasetUpload).filter(DatasetUpload.id == dataset_id).first()


def list_dataset_uploads(db: Session, limit: int = 50) -> list[DatasetUpload]:
    return db.query(DatasetUpload).order_by(DatasetUpload.created_at.desc()).limit(limit).all()


def delete_dataset_upload(db: Session, dataset_id: int) -> bool:
    upload = db.query(DatasetUpload).filter(DatasetUpload.id == dataset_id).first()
    if not upload:
        return False
    db.delete(upload)
    return True


def load_dataset_as_dataframe(db: Session, dataset_id: int) -> pd.DataFrame | None:
    rows = (
        db.query(StudentRecord)
        .filter(StudentRecord.dataset_id == dataset_id)
        .order_by(StudentRecord.row_number)
        .all()
    )
    if not rows:
        return None
    return pd.DataFrame([r.data for r in rows])


def get_latest_dataset_upload(db: Session) -> DatasetUpload | None:
    return db.query(DatasetUpload).order_by(DatasetUpload.created_at.desc()).first()


def update_dataset_quality(db: Session, dataset_id: int, quality_score: float, quality_report: dict) -> None:
    db.query(DatasetUpload).filter(DatasetUpload.id == dataset_id).update(
        {"quality_score": quality_score, "quality_report": quality_report, "status": "processed"}
    )


# ---------------------------------------------------------------------------
# ETL jobs
# ---------------------------------------------------------------------------

def create_etl_job(db: Session, dataset_upload_id: int | None = None) -> EtlJob:
    job = EtlJob(dataset_upload_id=dataset_upload_id, status="running", stage="extract")
    db.add(job)
    db.flush()
    return job


def update_etl_job(db: Session, job_id: int, **kwargs) -> None:
    db.query(EtlJob).filter(EtlJob.id == job_id).update(kwargs)


def complete_etl_job(db: Session, job_id: int, rows_loaded: int, rows_rejected: int = 0) -> None:
    now = datetime.now(timezone.utc)
    job = db.query(EtlJob).filter(EtlJob.id == job_id).first()
    if job:
        started = job.started_at.replace(tzinfo=timezone.utc) if job.started_at.tzinfo is None else job.started_at
        duration = (now - started).total_seconds()
        db.query(EtlJob).filter(EtlJob.id == job_id).update({
            "status": "completed",
            "stage": "done",
            "rows_loaded": rows_loaded,
            "rows_rejected": rows_rejected,
            "completed_at": now,
            "duration_seconds": duration,
        })


def fail_etl_job(db: Session, job_id: int, error_message: str) -> None:
    db.query(EtlJob).filter(EtlJob.id == job_id).update({
        "status": "failed",
        "error_message": error_message,
        "completed_at": datetime.now(timezone.utc),
    })


def list_etl_jobs(db: Session, limit: int = 50) -> list[EtlJob]:
    return db.query(EtlJob).order_by(EtlJob.started_at.desc()).limit(limit).all()


# ---------------------------------------------------------------------------
# KPI snapshots
# ---------------------------------------------------------------------------

def save_kpi_snapshot(db: Session, dataset_id: int, kpis: list[dict]) -> None:
    db.query(KpiSnapshot).filter(KpiSnapshot.dataset_id == dataset_id).delete()
    db.bulk_save_objects([
        KpiSnapshot(
            dataset_id=dataset_id,
            kpi_name=kpi["name"],
            kpi_value=kpi.get("value"),
            kpi_label=kpi.get("label"),
            kpi_dimension=kpi.get("dimension"),
            kpi_group=kpi.get("group"),
        )
        for kpi in kpis
    ])


def get_kpi_snapshot(db: Session, dataset_id: int) -> list[KpiSnapshot]:
    return db.query(KpiSnapshot).filter(KpiSnapshot.dataset_id == dataset_id).all()


def get_kpi_history(db: Session, kpi_name: str, limit: int = 50) -> list[KpiSnapshot]:
    return (
        db.query(KpiSnapshot)
        .filter(KpiSnapshot.kpi_name == kpi_name)
        .order_by(KpiSnapshot.computed_at.desc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Model runs
# ---------------------------------------------------------------------------

def save_model_run(
    db: Session,
    dataset_id: int,
    run_type: str,
    feature_columns: list[str],
    target_source: str | None,
    train_rows: int | None,
    test_rows: int | None,
    cv_folds: int | None,
    performance_summary: str | None,
    run_config: dict | None,
    results: list[dict],
) -> ModelRun:
    run = ModelRun(
        dataset_id=dataset_id,
        run_type=run_type,
        feature_columns=feature_columns,
        target_source=target_source,
        train_rows=train_rows,
        test_rows=test_rows,
        cv_folds=cv_folds,
        performance_summary=performance_summary,
        run_config=run_config,
    )
    db.add(run)
    db.flush()

    db.bulk_save_objects([
        ModelResult(
            run_id=run.id,
            model_name=r.get("model_name", ""),
            accuracy=r.get("accuracy"),
            precision=r.get("precision"),
            recall=r.get("recall"),
            f1_score=r.get("f1_score"),
            roc_auc=r.get("roc_auc"),
            average_precision=r.get("average_precision"),
            log_loss=r.get("log_loss"),
            brier_score=r.get("brier_score"),
            feature_importance=r.get("feature_importance"),
            cv_metrics=r.get("cv_metrics"),
            confusion_matrix=r.get("confusion_matrix"),
            probability_calibrated=r.get("probability_calibrated", False),
            summary=r.get("summary"),
        )
        for r in results
    ])
    return run


def list_model_runs(db: Session, dataset_id: int | None = None, limit: int = 50) -> list[ModelRun]:
    q = db.query(ModelRun)
    if dataset_id:
        q = q.filter(ModelRun.dataset_id == dataset_id)
    return q.order_by(ModelRun.created_at.desc()).limit(limit).all()


# ---------------------------------------------------------------------------
# Prediction events
# ---------------------------------------------------------------------------

def save_prediction_event(
    db: Session,
    model_name: str,
    input_features: dict,
    predicted_label: str,
    pass_probability: float,
    fail_probability: float,
    confidence_score: float,
    risk_level: str,
    recommendation: str,
    user_id: int | None = None,
    dataset_id: int | None = None,
    shap_values: dict | None = None,
) -> PredictionEvent:
    event = PredictionEvent(
        user_id=user_id,
        dataset_id=dataset_id,
        model_name=model_name,
        input_features=input_features,
        predicted_label=predicted_label,
        pass_probability=pass_probability,
        fail_probability=fail_probability,
        confidence_score=confidence_score,
        risk_level=risk_level,
        recommendation=recommendation,
        shap_values=shap_values,
    )
    db.add(event)
    db.flush()
    return event


def list_prediction_events(db: Session, limit: int = 100) -> list[PredictionEvent]:
    return db.query(PredictionEvent).order_by(PredictionEvent.created_at.desc()).limit(limit).all()


def prediction_events_as_dataframe(db: Session, limit: int = 200) -> pd.DataFrame:
    events = list_prediction_events(db, limit=limit)
    if not events:
        return pd.DataFrame()
    rows = []
    for e in events:
        row = {
            "id": e.id,
            "model_name": e.model_name,
            "predicted_label": e.predicted_label,
            "pass_probability": e.pass_probability,
            "fail_probability": e.fail_probability,
            "confidence_score": e.confidence_score,
            "risk_level": e.risk_level,
            "recommendation": e.recommendation,
            "created_at": e.created_at,
        }
        row.update(e.input_features or {})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

def save_forecast(
    db: Session,
    dataset_id: int,
    forecast_type: str,
    method: str,
    horizon_days: int,
    data_points_used: int,
    predictions: dict,
    model_params: dict | None = None,
) -> ForecastResult:
    fr = ForecastResult(
        dataset_id=dataset_id,
        forecast_type=forecast_type,
        method=method,
        horizon_days=horizon_days,
        data_points_used=data_points_used,
        predictions=predictions,
        model_params=model_params,
    )
    db.add(fr)
    db.flush()
    return fr


def get_latest_forecast(db: Session, dataset_id: int, forecast_type: str) -> ForecastResult | None:
    return (
        db.query(ForecastResult)
        .filter(ForecastResult.dataset_id == dataset_id, ForecastResult.forecast_type == forecast_type)
        .order_by(ForecastResult.created_at.desc())
        .first()
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def create_report(
    db: Session,
    report_type: str,
    report_name: str,
    format: str = "pdf",
    dataset_id: int | None = None,
    generated_by: int | None = None,
) -> Report:
    r = Report(
        dataset_id=dataset_id,
        generated_by=generated_by,
        report_type=report_type,
        report_name=report_name,
        format=format,
        status="pending",
    )
    db.add(r)
    db.flush()
    return r


def complete_report(db: Session, report_id: int, file_path: str, file_size_bytes: int) -> None:
    db.query(Report).filter(Report.id == report_id).update({
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "status": "ready",
    })


def list_reports(db: Session, limit: int = 50) -> list[Report]:
    return db.query(Report).order_by(Report.created_at.desc()).limit(limit).all()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def write_audit_log(
    db: Session,
    action: str,
    user_id: int | None = None,
    user_email: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_summary: str | None = None,
    response_status: int | None = None,
    duration_ms: int | None = None,
) -> AuditLog:
    log = AuditLog(
        user_id=user_id,
        user_email=user_email,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        ip_address=ip_address,
        user_agent=user_agent,
        request_summary=request_summary,
        response_status=response_status,
        duration_ms=duration_ms,
    )
    db.add(log)
    db.flush()
    return log


def list_audit_logs(db: Session, limit: int = 200, action: str | None = None) -> list[AuditLog]:
    q = db.query(AuditLog)
    if action:
        q = q.filter(AuditLog.action == action)
    return q.order_by(AuditLog.created_at.desc()).limit(limit).all()


def audit_logs_as_dataframe(db: Session, limit: int = 500) -> pd.DataFrame:
    logs = list_audit_logs(db, limit=limit)
    if not logs:
        return pd.DataFrame()
    return pd.DataFrame([{
        "id": l.id,
        "user_email": l.user_email,
        "action": l.action,
        "resource_type": l.resource_type,
        "resource_id": l.resource_id,
        "ip_address": l.ip_address,
        "response_status": l.response_status,
        "duration_ms": l.duration_ms,
        "created_at": l.created_at,
    } for l in logs])
