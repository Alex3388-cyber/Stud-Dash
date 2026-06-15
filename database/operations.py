"""SQLite operations for saved datasets and prediction history.

All SQL statements use parameters instead of string interpolation. This keeps
database writes safe when file names, column names, or recommendation text come
from user-controlled input.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from database.connection import get_connection, initialize_database


def dataframe_to_json_records(data: pd.DataFrame) -> list[str]:
    """Convert dataframe rows to JSON strings that SQLite can store."""
    serializable_data = data.astype(object).where(pd.notna(data), None)
    records = serializable_data.to_dict(orient="records")
    return [json.dumps(record, default=str) for record in records]


def save_uploaded_dataset(file_name: str, data: pd.DataFrame) -> int:
    """Save dataset metadata and row snapshots, returning the dataset ID."""
    initialize_database()
    column_names = json.dumps(list(data.columns))
    row_payloads = dataframe_to_json_records(data)

    with get_connection() as connection:
        # First save dataset-level metadata so each row can reference this upload.
        cursor = connection.execute(
            """
            INSERT INTO dataset_uploads (file_name, row_count, column_count, column_names)
            VALUES (?, ?, ?, ?)
            """,
            (file_name, len(data), len(data.columns), column_names),
        )
        dataset_id = int(cursor.lastrowid)

        # Store each source row as JSON. This flexible approach supports uploaded
        # datasets with different schemas without creating new SQL columns.
        connection.executemany(
            """
            INSERT INTO dataset_rows (dataset_id, row_number, row_data)
            VALUES (?, ?, ?)
            """,
            [(dataset_id, row_number, row_data) for row_number, row_data in enumerate(row_payloads, start=1)],
        )

    return dataset_id


def save_prediction_history(
    dataset_name: str,
    model_name: str,
    inputs: dict[str, float],
    prediction: Any,
) -> int:
    """Persist one prediction request and its model output."""
    initialize_database()

    with get_connection() as connection:
        # Prediction history stores both the input values and the model response
        # so users can audit what produced each saved Pass/Fail result.
        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                dataset_name,
                model_name,
                study_time,
                absences,
                failures,
                previous_grade_1,
                previous_grade_2,
                predicted_label,
                pass_probability,
                fail_probability,
                risk_level,
                recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                model_name,
                inputs["study_time"],
                inputs["absences"],
                inputs["failures"],
                inputs["previous_grade_1"],
                inputs["previous_grade_2"],
                prediction.predicted_label,
                prediction.pass_probability,
                prediction.fail_probability,
                prediction.risk_level,
                prediction.recommendation,
            ),
        )
        return int(cursor.lastrowid)


def fetch_dataset_uploads(limit: int = 25) -> pd.DataFrame:
    """Retrieve recent uploaded dataset records."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                dataset_id,
                file_name,
                row_count,
                column_count,
                column_names,
                created_at
            FROM dataset_uploads
            ORDER BY created_at DESC, dataset_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    data = pd.DataFrame([dict(row) for row in rows])
    if data.empty:
        return data

    data["column_names"] = data["column_names"].apply(lambda value: ", ".join(json.loads(value)))
    return data


def fetch_dataset_rows(dataset_id: int, limit: int = 100) -> pd.DataFrame:
    """Retrieve saved row snapshots for a dataset upload."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT row_number, row_data, created_at
            FROM dataset_rows
            WHERE dataset_id = ?
            ORDER BY row_number ASC
            LIMIT ?
            """,
            (dataset_id, limit),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    decoded_rows = []
    for row in rows:
        record = json.loads(row["row_data"])
        record["row_number"] = row["row_number"]
        record["created_at"] = row["created_at"]
        decoded_rows.append(record)
    return pd.DataFrame(decoded_rows)


def fetch_saved_dataset(dataset_id: int) -> tuple[pd.DataFrame | None, str | None]:
    """Rebuild one saved dataset from SQLite row snapshots."""
    initialize_database()
    with get_connection() as connection:
        dataset_row = connection.execute(
            """
            SELECT file_name, column_names
            FROM dataset_uploads
            WHERE dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()
        if dataset_row is None:
            return None, None

        rows = connection.execute(
            """
            SELECT row_data
            FROM dataset_rows
            WHERE dataset_id = ?
            ORDER BY row_number ASC
            """,
            (dataset_id,),
        ).fetchall()

    records = [json.loads(row["row_data"]) for row in rows]
    data = pd.DataFrame(records)
    stored_columns = json.loads(dataset_row["column_names"])
    ordered_columns = [column for column in stored_columns if column in data.columns]
    ordered_columns.extend(column for column in data.columns if column not in ordered_columns)
    if ordered_columns:
        data = data.reindex(columns=ordered_columns)
    return data, str(dataset_row["file_name"])


def fetch_latest_saved_dataset() -> tuple[pd.DataFrame | None, str | None, int | None]:
    """Return the most recent saved upload so the dashboard can restore it."""
    initialize_database()
    with get_connection() as connection:
        latest_row = connection.execute(
            """
            SELECT dataset_id
            FROM dataset_uploads
            ORDER BY created_at DESC, dataset_id DESC
            LIMIT 1
            """
        ).fetchone()

    if latest_row is None:
        return None, None, None

    dataset_id = int(latest_row["dataset_id"])
    data, file_name = fetch_saved_dataset(dataset_id)
    return data, file_name, dataset_id


def fetch_prediction_history(limit: int = 50) -> pd.DataFrame:
    """Retrieve recent saved prediction records."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                prediction_id,
                dataset_name,
                model_name,
                study_time,
                absences,
                failures,
                previous_grade_1,
                previous_grade_2,
                predicted_label,
                pass_probability,
                fail_probability,
                risk_level,
                recommendation,
                created_at
            FROM prediction_history
            ORDER BY created_at DESC, prediction_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def count_saved_records() -> dict[str, int]:
    """Return quick database counts for dashboard display."""
    initialize_database()
    with get_connection() as connection:
        datasets = connection.execute("SELECT COUNT(*) AS total FROM dataset_uploads").fetchone()["total"]
        rows = connection.execute("SELECT COUNT(*) AS total FROM dataset_rows").fetchone()["total"]
        predictions = connection.execute("SELECT COUNT(*) AS total FROM prediction_history").fetchone()["total"]
    return {"datasets": datasets, "rows": rows, "predictions": predictions}


def clear_saved_datasets() -> dict[str, int]:
    """Delete all saved dataset uploads and their stored row snapshots."""
    initialize_database()
    with get_connection() as connection:
        dataset_count = int(connection.execute("SELECT COUNT(*) AS total FROM dataset_uploads").fetchone()["total"])
        row_count = int(connection.execute("SELECT COUNT(*) AS total FROM dataset_rows").fetchone()["total"])
        connection.execute("DELETE FROM dataset_uploads")
    return {"datasets_deleted": dataset_count, "rows_deleted": row_count}


def clear_prediction_history() -> int:
    """Delete all saved prediction history records."""
    initialize_database()
    with get_connection() as connection:
        prediction_count = int(connection.execute("SELECT COUNT(*) AS total FROM prediction_history").fetchone()["total"])
        connection.execute("DELETE FROM prediction_history")
    return prediction_count


def clear_all_saved_records() -> dict[str, int]:
    """Delete saved datasets, saved dataset rows, and prediction history."""
    dataset_result = clear_saved_datasets()
    prediction_count = clear_prediction_history()
    return {
        "datasets_deleted": dataset_result["datasets_deleted"],
        "rows_deleted": dataset_result["rows_deleted"],
        "predictions_deleted": prediction_count,
    }


# ---------------------------------------------------------------------------
# ETL monitoring (Phase 2)
# ---------------------------------------------------------------------------

def log_etl_job(
    dataset_name: str,
    stage: str,
    rows_in: int,
    rows_out: int,
    rows_rejected: int,
    duration_ms: int,
    status: str = "completed",
    error_message: str | None = None,
) -> int:
    """Record one ETL pipeline stage execution."""
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO etl_jobs
                (dataset_name, stage, rows_in, rows_out, rows_rejected, duration_ms, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_name, stage, rows_in, rows_out, rows_rejected, duration_ms, status, error_message),
        )
        return int(cursor.lastrowid)


def fetch_etl_jobs(limit: int = 50) -> pd.DataFrame:
    """Return recent ETL job records."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT job_id, dataset_name, stage, rows_in, rows_out, rows_rejected,
                   duration_ms, status, error_message, started_at
            FROM etl_jobs
            ORDER BY started_at DESC, job_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Audit log (Phase 10)
# ---------------------------------------------------------------------------

def log_audit_event(
    event_type: str,
    entity_name: str | None = None,
    detail: str | None = None,
    rows_affected: int | None = None,
) -> int:
    """Record one audit event."""
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO audit_events (event_type, entity_name, detail, rows_affected)
            VALUES (?, ?, ?, ?)
            """,
            (event_type, entity_name, detail, rows_affected),
        )
        return int(cursor.lastrowid)


def fetch_audit_events(limit: int = 200) -> pd.DataFrame:
    """Return recent audit events."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT event_id, event_type, entity_name, detail, rows_affected, created_at
            FROM audit_events
            ORDER BY created_at DESC, event_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])
