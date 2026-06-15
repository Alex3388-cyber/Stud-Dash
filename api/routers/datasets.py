"""Dataset management router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user, require_steward
from api.schemas.dataset import DatasetList, DatasetMeta
from database.models import User
from database.pg_operations import delete_dataset_upload, get_dataset_upload, list_dataset_uploads
from database.pg_operations import write_audit_log

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetList)
def list_datasets(limit: int = 50, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    uploads = list_dataset_uploads(db, limit=limit)
    return DatasetList(datasets=uploads, total=len(uploads))


@router.get("/{dataset_id}", response_model=DatasetMeta)
def get_dataset(dataset_id: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    upload = get_dataset_upload(db, dataset_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return upload


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_steward),
):
    ok = delete_dataset_upload(db, dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Dataset not found")
    write_audit_log(
        db, action="DELETE_DATASET", user_id=current_user.id,
        user_email=current_user.email, resource_type="dataset", resource_id=str(dataset_id),
    )
