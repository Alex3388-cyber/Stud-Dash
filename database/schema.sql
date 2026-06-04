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
