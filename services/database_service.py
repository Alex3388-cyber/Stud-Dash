"""Service wrapper for SQLite operations used by the dashboard."""

from __future__ import annotations

from database.connection import initialize_database
from database.operations import (
    clear_all_saved_records,
    clear_prediction_history,
    clear_saved_datasets,
    count_saved_records,
    fetch_audit_events,
    fetch_dataset_rows,
    fetch_dataset_uploads,
    fetch_etl_jobs,
    fetch_prediction_history,
    log_audit_event,
    log_etl_job,
    save_prediction_history,
    save_uploaded_dataset,
)


def initialize_storage() -> None:
    """Initialize SQLite storage."""
    initialize_database()


def save_dataset_upload(dataset_name, data):
    """Persist an uploaded dataset."""
    return save_uploaded_dataset(dataset_name, data)


def save_prediction(dataset_name, model_name, inputs, prediction):
    """Persist a prediction record."""
    return save_prediction_history(
        dataset_name=dataset_name,
        model_name=model_name,
        inputs=inputs,
        prediction=prediction,
    )


def get_record_counts():
    """Return dashboard SQLite record counts."""
    return count_saved_records()


def get_dataset_uploads():
    """Return saved dataset uploads."""
    return fetch_dataset_uploads()


def get_dataset_rows(dataset_id: int):
    """Return saved rows for one uploaded dataset."""
    return fetch_dataset_rows(dataset_id)


def get_prediction_history(limit: int = 50):
    """Return saved prediction history."""
    return fetch_prediction_history(limit=limit)


def clear_saved_dataset_records():
    """Clear saved dataset uploads and row snapshots."""
    return clear_saved_datasets()


def clear_saved_prediction_records():
    """Clear saved prediction history."""
    return clear_prediction_history()


def clear_all_records():
    """Clear all persisted dashboard records."""
    return clear_all_saved_records()


def record_etl_job(dataset_name, stage, rows_in, rows_out, rows_rejected=0, duration_ms=0, status="completed", error=None):
    """Log an ETL pipeline stage."""
    return log_etl_job(dataset_name, stage, rows_in, rows_out, rows_rejected, duration_ms, status, error)


def get_etl_jobs(limit: int = 50):
    """Fetch ETL job history."""
    return fetch_etl_jobs(limit)


def record_audit_event(event_type: str, entity_name: str | None = None, detail: str | None = None, rows_affected: int | None = None):
    """Log an audit event."""
    return log_audit_event(event_type, entity_name, detail, rows_affected)


def get_audit_events(limit: int = 200):
    """Fetch audit log events."""
    return fetch_audit_events(limit)
