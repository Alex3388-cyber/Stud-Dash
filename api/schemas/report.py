from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class ReportRequest(BaseModel):
    report_type: str
    dataset_id: int | None = None
    format: str = "pdf"
    options: dict | None = None


class ReportOut(BaseModel):
    id: int
    report_type: str
    report_name: str
    format: str
    status: str
    file_size_bytes: int | None
    created_at: datetime

    model_config = {"from_attributes": True}
