"""KPI router — materialised snapshots and history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from api.schemas.kpi import KpiHistoryPoint, KpiSnapshotOut
from database.models import User
from database.pg_operations import get_dataset_upload, get_kpi_history, get_kpi_snapshot
from services.kpi_engine import compute_and_persist_kpis

router = APIRouter(prefix="/kpi", tags=["kpi"])


@router.get("/snapshot/{dataset_id}", response_model=KpiSnapshotOut)
def kpi_snapshot(dataset_id: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    upload = get_dataset_upload(db, dataset_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Dataset not found")

    rows = get_kpi_snapshot(db, dataset_id)
    if not rows:
        compute_and_persist_kpis(db, dataset_id)
        rows = get_kpi_snapshot(db, dataset_id)

    if not rows:
        raise HTTPException(status_code=404, detail="No KPIs computed for this dataset")

    return KpiSnapshotOut(
        dataset_id=dataset_id,
        kpis=[{
            "name": r.kpi_name,
            "value": r.kpi_value,
            "label": r.kpi_label,
            "dimension": r.kpi_dimension,
            "group": r.kpi_group,
            "computed_at": r.computed_at,
        } for r in rows],
        computed_at=rows[0].computed_at,
    )


@router.post("/compute/{dataset_id}", status_code=202)
def trigger_kpi_compute(dataset_id: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    upload = get_dataset_upload(db, dataset_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Dataset not found")
    compute_and_persist_kpis(db, dataset_id)
    return {"status": "computed", "dataset_id": dataset_id}


@router.get("/history/{kpi_name}", response_model=list[KpiHistoryPoint])
def kpi_history(kpi_name: str, limit: int = 50, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    rows = get_kpi_history(db, kpi_name, limit=limit)
    return [{"kpi_name": r.kpi_name, "kpi_value": r.kpi_value, "kpi_label": r.kpi_label, "computed_at": r.computed_at} for r in rows]
