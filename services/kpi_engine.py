"""KPI generation engine — computes and materialises KPIs from a dataset."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from database.pg_operations import load_dataset_as_dataframe, save_kpi_snapshot
from services.analytics_service import (
    build_home_kpis,
    calculate_average_attendance,
    calculate_average_score,
    calculate_pass_fail_kpis,
    calculate_row_average_scores,
    get_pass_fail_target,
    get_total_students,
)
from services.dataset_service import get_score_columns, get_attendance_column
from utilities.schema_mapping import build_auto_schema_mapping


def _build_kpi_list(data: pd.DataFrame) -> list[dict]:
    """Compute KPIs from a DataFrame and return them as a list of dicts for DB storage."""
    schema = build_auto_schema_mapping(data)

    score_cols_all = schema.get("score_columns", [])
    score_cols = [c for c in score_cols_all if c in data.columns]
    attendance_col = schema.get("attendance_column")
    if attendance_col and attendance_col not in data.columns:
        attendance_col = None

    avg_score = calculate_average_score(data, score_cols)
    avg_attendance = calculate_average_attendance(data, attendance_col)
    total_students = get_total_students(data)
    target, _col, target_source = get_pass_fail_target(data, score_cols)
    pf = calculate_pass_fail_kpis(target)
    row_scores = calculate_row_average_scores(data, score_cols)

    kpis = [
        {"name": "total_students", "value": float(total_students), "label": str(total_students), "group": "enrolment"},
        {"name": "pass_rate", "value": pf.get("pass_rate"), "label": f"{pf.get('pass_rate', 0):.1f}%" if pf.get("pass_rate") is not None else "N/A", "group": "performance"},
        {"name": "fail_rate", "value": pf.get("fail_rate"), "label": f"{pf.get('fail_rate', 0):.1f}%" if pf.get("fail_rate") is not None else "N/A", "group": "performance"},
        {"name": "pass_count", "value": float(pf["pass_count"]) if pf.get("pass_count") is not None else None, "label": str(pf.get("pass_count", "N/A")), "group": "performance"},
        {"name": "fail_count", "value": float(pf["fail_count"]) if pf.get("fail_count") is not None else None, "label": str(pf.get("fail_count", "N/A")), "group": "performance"},
        {"name": "average_score", "value": avg_score, "label": f"{avg_score:.1f}%" if avg_score is not None else "N/A", "group": "performance"},
        {"name": "average_attendance", "value": avg_attendance, "label": f"{avg_attendance:.1f}%" if avg_attendance is not None else "N/A", "group": "attendance"},
        {"name": "score_columns_count", "value": float(len(score_cols)), "label": str(len(score_cols)), "group": "metadata"},
    ]

    if not row_scores.empty:
        at_risk = int((row_scores < 60).sum())
        high_perf = int((row_scores >= 80).sum())
        kpis.append({"name": "at_risk_count", "value": float(at_risk), "label": str(at_risk), "group": "risk"})
        kpis.append({"name": "high_performers_count", "value": float(high_perf), "label": str(high_perf), "group": "risk"})
        kpis.append({"name": "average_row_score", "value": float(row_scores.mean()), "label": f"{row_scores.mean():.1f}", "group": "performance"})

    return kpis


def compute_and_persist_kpis(db: Session, dataset_id: int) -> list[dict]:
    """Load dataset from DB, compute all KPIs, and persist them."""
    data = load_dataset_as_dataframe(db, dataset_id)
    if data is None or data.empty:
        return []

    kpis = _build_kpi_list(data)
    save_kpi_snapshot(db, dataset_id, kpis)
    return kpis


def get_kpis_for_dataset(db: Session, dataset_id: int) -> list[dict]:
    """Return computed KPIs, computing them if not yet materialised."""
    from database.pg_operations import get_kpi_snapshot
    rows = get_kpi_snapshot(db, dataset_id)
    if rows:
        return [{"name": r.kpi_name, "value": r.kpi_value, "label": r.kpi_label, "group": r.kpi_group} for r in rows]
    return compute_and_persist_kpis(db, dataset_id)
