from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class KpiCardOut(BaseModel):
    name: str
    value: float | None
    label: str | None
    dimension: str | None
    group: str | None
    computed_at: datetime

    model_config = {"from_attributes": True}


class KpiSnapshotOut(BaseModel):
    dataset_id: int
    kpis: list[KpiCardOut]
    computed_at: datetime


class KpiHistoryPoint(BaseModel):
    kpi_name: str
    kpi_value: float | None
    kpi_label: str | None
    computed_at: datetime
