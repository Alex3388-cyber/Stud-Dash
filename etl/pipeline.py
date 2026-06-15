"""ETL pipeline orchestrator — extract → validate → transform → load."""

from __future__ import annotations

from sqlalchemy.orm import Session

from database.models import DatasetUpload, EtlJob
from database.pg_operations import (
    complete_etl_job,
    create_etl_job,
    fail_etl_job,
    save_dataset_upload,
    update_dataset_quality,
    update_etl_job,
)
from etl.extractor import extract
from etl.transformer import transform
from etl.validator import validate
from services.kpi_engine import compute_and_persist_kpis


def run_etl_pipeline(
    db: Session,
    file_name: str,
    file_content: bytes,
    uploaded_by: int | None = None,
) -> tuple[DatasetUpload, EtlJob]:
    """Run the full ETL pipeline synchronously and return the upload + job records."""
    job = create_etl_job(db)

    try:
        # Extract
        update_etl_job(db, job.id, stage="extract")
        raw_data = extract(file_name, file_content)
        update_etl_job(db, job.id, rows_extracted=len(raw_data))

        # Validate
        update_etl_job(db, job.id, stage="validate")
        quality_report = validate(raw_data)
        update_etl_job(db, job.id, rows_validated=quality_report.total_rows)

        # Transform
        update_etl_job(db, job.id, stage="transform")
        clean_data = transform(raw_data)
        rejected = len(raw_data) - len(clean_data)

        # Load
        update_etl_job(db, job.id, stage="load")
        upload = save_dataset_upload(
            db=db,
            file_name=file_name,
            data=clean_data,
            uploaded_by=uploaded_by,
            file_size_bytes=len(file_content),
        )

        update_dataset_quality(db, upload.id, quality_report.overall_score, quality_report.to_dict())
        update_etl_job(db, job.id, dataset_upload_id=upload.id)
        complete_etl_job(db, job.id, rows_loaded=len(clean_data), rows_rejected=rejected)

        # Materialise KPIs immediately after load
        try:
            compute_and_persist_kpis(db, upload.id)
        except Exception:
            pass

        return upload, job

    except Exception as exc:
        fail_etl_job(db, job.id, error_message=str(exc))
        raise
