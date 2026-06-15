"""ETL router — file upload, job tracking, data quality."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user, get_optional_user
from api.schemas.dataset import DatasetMeta, EtlJobOut
from database.models import User
from database.pg_operations import get_dataset_upload, list_dataset_uploads, list_etl_jobs
from etl.pipeline import run_etl_pipeline

router = APIRouter(prefix="/etl", tags=["etl"])


@router.post("/upload", response_model=DatasetMeta, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    upload, _job = run_etl_pipeline(
        db=db,
        file_name=file.filename or "upload",
        file_content=content,
        uploaded_by=current_user.id if current_user else None,
    )
    return upload


@router.get("/jobs", response_model=list[EtlJobOut])
def get_jobs(limit: int = 50, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    return list_etl_jobs(db, limit=limit)


@router.get("/jobs/{job_id}", response_model=EtlJobOut)
def get_job(job_id: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    from database.models import EtlJob
    job = db.query(EtlJob).filter(EtlJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
