from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class DatasetMeta(BaseModel):
    id: int
    file_name: str
    row_count: int
    column_count: int
    column_names: list[str]
    quality_score: float | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetList(BaseModel):
    datasets: list[DatasetMeta]
    total: int


class EtlJobOut(BaseModel):
    id: int
    dataset_upload_id: int | None
    status: str
    stage: str
    rows_extracted: int
    rows_validated: int
    rows_loaded: int
    rows_rejected: int
    error_message: str | None
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None

    model_config = {"from_attributes": True}
