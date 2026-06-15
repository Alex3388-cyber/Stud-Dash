-- SQLite schema for dashboard persistence.
-- The tables are intentionally flexible so uploaded datasets with different
-- academic schemas can be stored without changing the database structure.

CREATE TABLE IF NOT EXISTS dataset_uploads (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    column_names TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset_rows (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    row_number INTEGER NOT NULL,
    row_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES dataset_uploads(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prediction_history (
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

-- ETL monitoring: one row per pipeline run (Phase 2)
CREATE TABLE IF NOT EXISTS etl_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    stage TEXT NOT NULL,
    rows_in INTEGER,
    rows_out INTEGER,
    rows_rejected INTEGER DEFAULT 0,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'completed',
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log: every significant user action (Phase 10)
CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_name TEXT,
    detail TEXT,
    rows_affected INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
