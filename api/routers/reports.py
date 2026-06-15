"""Report generation router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from api.schemas.report import ReportOut, ReportRequest
from database.models import Report, User
from database.pg_operations import complete_report, create_report, list_reports, load_dataset_as_dataframe
from services.report_service import build_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/generate", response_model=ReportOut, status_code=201)
def generate_report(
    body: ReportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    data = load_dataset_as_dataframe(db, body.dataset_id) if body.dataset_id else None

    report_record = create_report(
        db=db,
        report_type=body.report_type,
        report_name=f"{body.report_type.replace('_', ' ').title()} Report",
        format=body.format,
        dataset_id=body.dataset_id,
        generated_by=current_user.id,
    )

    file_path, file_size = build_report(
        report_id=report_record.id,
        report_type=body.report_type,
        format=body.format,
        data=data,
        options=body.options or {},
    )

    complete_report(db, report_record.id, file_path, file_size)
    return db.query(Report).filter(Report.id == report_record.id).first()


@router.get("", response_model=list[ReportOut])
def list_all_reports(limit: int = 50, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    return list_reports(db, limit=limit)


@router.get("/{report_id}/download")
def download_report(report_id: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.status != "ready" or not report.file_path:
        raise HTTPException(status_code=400, detail="Report is not ready yet")
    return FileResponse(report.file_path, filename=f"{report.report_name}.{report.format}")
