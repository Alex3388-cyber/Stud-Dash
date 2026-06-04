"""SQLite connection helpers for the dashboard."""

from contextlib import contextmanager
import sqlite3
from pathlib import Path
from typing import Iterator, Iterable


DATABASE_PATH = Path(__file__).resolve().parent / "student_performance.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def open_connection(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Create a raw SQLite connection with row access by column name."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    # Foreign keys are disabled by default in SQLite, so they are explicitly
    # enabled for each connection to keep dataset row relationships valid.
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def get_connection(database_path: Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection and always close it after use."""
    connection = open_connection(database_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: Path = DATABASE_PATH, schema_path: Path = SCHEMA_PATH) -> None:
    """Create database tables from the schema file when they do not already exist."""
    with get_connection(database_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        synchronize_prediction_history_schema(connection)


def synchronize_prediction_history_schema(connection: sqlite3.Connection) -> None:
    """Keep the prediction history table aligned with the active console schema.

    Older project versions stored attendance, assignment score, study hours, and
    midsem score columns. The live AI console now stores UCI-style inputs
    instead. When an older table is detected, it is archived and a fresh table
    is created because the new insert/query paths require the new field names.
    """
    existing_table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'prediction_history'"
    ).fetchone()
    if not existing_table:
        return

    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(prediction_history)").fetchall()
    }
    expected_columns = {
        "prediction_id",
        "dataset_name",
        "model_name",
        "study_time",
        "absences",
        "failures",
        "previous_grade_1",
        "previous_grade_2",
        "predicted_label",
        "pass_probability",
        "fail_probability",
        "risk_level",
        "recommendation",
        "created_at",
    }
    if expected_columns.issubset(columns):
        return

    archive_name = "prediction_history_legacy"
    suffix = 1
    while connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (archive_name,),
    ).fetchone():
        suffix += 1
        archive_name = f"prediction_history_legacy_{suffix}"

    connection.execute(f"ALTER TABLE prediction_history RENAME TO {archive_name}")
    connection.executescript(
        """
        CREATE TABLE prediction_history (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            model_name TEXT NOT NULL,
            study_time REAL NOT NULL,
            absences REAL NOT NULL,
            failures REAL NOT NULL,
            previous_grade_1 REAL NOT NULL,
            previous_grade_2 REAL NOT NULL,
            predicted_label TEXT NOT NULL,
            pass_probability REAL NOT NULL,
            fail_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def execute_many(query: str, rows: Iterable[tuple], database_path: Path = DATABASE_PATH) -> None:
    """Execute a parameterized query for multiple rows."""
    with get_connection(database_path) as connection:
        connection.executemany(query, rows)
