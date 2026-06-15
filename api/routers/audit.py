"""Audit log router — admin only."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_db, require_admin
from database.models import User
from database.pg_operations import list_audit_logs

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditLogOut:
    pass


@router.get("/logs")
def get_audit_logs(
    limit: int = 200,
    action: str | None = None,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    logs = list_audit_logs(db, limit=limit, action=action)
    return [
        {
            "id": l.id,
            "user_email": l.user_email,
            "action": l.action,
            "resource_type": l.resource_type,
            "resource_id": l.resource_id,
            "ip_address": l.ip_address,
            "response_status": l.response_status,
            "duration_ms": l.duration_ms,
            "created_at": l.created_at,
        }
        for l in logs
    ]
