"""SQLAlchemy ORM models for the Stud-Dash EDW.

Works with SQLite (dev) and PostgreSQL (prod) via DATABASE_URL.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Auth layer
# ---------------------------------------------------------------------------

class UserRole(str, enum.Enum):
    admin = "admin"
    data_steward = "data_steward"
    analyst = "analyst"
    viewer = "viewer"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.viewer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime)

    sessions: Mapped[list[Session]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    audit_logs: Mapped[list[AuditLog]] = relationship("AuditLog", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)
    refresh_token: Mapped[str | None] = mapped_column(String(512), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship("User", back_populates="sessions")


# ---------------------------------------------------------------------------
# Staging layer — raw ingest
# ---------------------------------------------------------------------------

class DatasetUpload(Base):
    """Registry of every file uploaded to the platform."""
    __tablename__ = "dataset_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    uploaded_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_names: Mapped[dict] = mapped_column(JSON, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    quality_score: Mapped[float | None] = mapped_column(Float)
    quality_report: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    etl_job: Mapped[EtlJob | None] = relationship("EtlJob", back_populates="dataset_upload", uselist=False)
    student_records: Mapped[list[StudentRecord]] = relationship("StudentRecord", back_populates="dataset_upload", cascade="all, delete-orphan")
    kpi_snapshots: Mapped[list[KpiSnapshot]] = relationship("KpiSnapshot", back_populates="dataset_upload", cascade="all, delete-orphan")
    model_runs: Mapped[list[ModelRun]] = relationship("ModelRun", back_populates="dataset_upload", cascade="all, delete-orphan")


class StudentRecord(Base):
    """One normalised student record row per uploaded dataset row."""
    __tablename__ = "student_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("dataset_uploads.id", ondelete="CASCADE"), nullable=False, index=True)
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    dataset_upload: Mapped[DatasetUpload] = relationship("DatasetUpload", back_populates="student_records")

    __table_args__ = (Index("ix_student_records_dataset_row", "dataset_id", "row_number"),)


# ---------------------------------------------------------------------------
# ETL layer
# ---------------------------------------------------------------------------

class EtlJob(Base):
    __tablename__ = "etl_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_upload_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dataset_uploads.id", ondelete="SET NULL"), index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    stage: Mapped[str] = mapped_column(String(50), default="extract", nullable=False)
    rows_extracted: Mapped[int] = mapped_column(Integer, default=0)
    rows_validated: Mapped[int] = mapped_column(Integer, default=0)
    rows_loaded: Mapped[int] = mapped_column(Integer, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    dataset_upload: Mapped[DatasetUpload | None] = relationship("DatasetUpload", back_populates="etl_job")


# ---------------------------------------------------------------------------
# KPI warehouse layer
# ---------------------------------------------------------------------------

class KpiSnapshot(Base):
    """Materialised KPI value per dataset, dimension, and time."""
    __tablename__ = "kpi_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("dataset_uploads.id", ondelete="CASCADE"), nullable=False, index=True)
    kpi_name: Mapped[str] = mapped_column(String(100), nullable=False)
    kpi_value: Mapped[float | None] = mapped_column(Float)
    kpi_label: Mapped[str | None] = mapped_column(String(255))
    kpi_dimension: Mapped[str | None] = mapped_column(String(100))
    kpi_group: Mapped[str | None] = mapped_column(String(100))
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    dataset_upload: Mapped[DatasetUpload] = relationship("DatasetUpload", back_populates="kpi_snapshots")

    __table_args__ = (Index("ix_kpi_snapshots_dataset_name", "dataset_id", "kpi_name"),)


# ---------------------------------------------------------------------------
# ML model layer
# ---------------------------------------------------------------------------

class ModelRun(Base):
    """Metadata for every classification or clustering run."""
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("dataset_uploads.id", ondelete="CASCADE"), nullable=False, index=True)
    run_type: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_columns: Mapped[dict] = mapped_column(JSON, nullable=False)
    target_source: Mapped[str | None] = mapped_column(String(255))
    train_rows: Mapped[int | None] = mapped_column(Integer)
    test_rows: Mapped[int | None] = mapped_column(Integer)
    cv_folds: Mapped[int | None] = mapped_column(Integer)
    performance_summary: Mapped[str | None] = mapped_column(Text)
    run_config: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    dataset_upload: Mapped[DatasetUpload] = relationship("DatasetUpload", back_populates="model_runs")
    model_results: Mapped[list[ModelResult]] = relationship("ModelResult", back_populates="model_run", cascade="all, delete-orphan")


class ModelResult(Base):
    """Per-model metrics within a training run."""
    __tablename__ = "model_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    roc_auc: Mapped[float | None] = mapped_column(Float)
    average_precision: Mapped[float | None] = mapped_column(Float)
    log_loss: Mapped[float | None] = mapped_column(Float)
    brier_score: Mapped[float | None] = mapped_column(Float)
    feature_importance: Mapped[dict | None] = mapped_column(JSON)
    cv_metrics: Mapped[dict | None] = mapped_column(JSON)
    confusion_matrix: Mapped[dict | None] = mapped_column(JSON)
    probability_calibrated: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[str | None] = mapped_column(Text)

    model_run: Mapped[ModelRun] = relationship("ModelRun", back_populates="model_results")


# ---------------------------------------------------------------------------
# Prediction events
# ---------------------------------------------------------------------------

class PredictionEvent(Base):
    """Every live prediction request logged for audit and analytics."""
    __tablename__ = "prediction_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    dataset_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dataset_uploads.id", ondelete="SET NULL"))
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(50), nullable=False)
    pass_probability: Mapped[float] = mapped_column(Float, nullable=False)
    fail_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(50), nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    shap_values: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class ForecastResult(Base):
    """Stored forecasting run output."""
    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("dataset_uploads.id", ondelete="CASCADE"), nullable=False, index=True)
    forecast_type: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    data_points_used: Mapped[int] = mapped_column(Integer, nullable=False)
    predictions: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_params: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dataset_uploads.id", ondelete="SET NULL"))
    generated_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    report_type: Mapped[str] = mapped_column(String(100), nullable=False)
    report_name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(20), default="pdf")
    file_path: Mapped[str | None] = mapped_column(String(512))
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), index=True)
    user_email: Mapped[str | None] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(100))
    resource_id: Mapped[str | None] = mapped_column(String(100))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(512))
    request_summary: Mapped[str | None] = mapped_column(Text)
    response_status: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    user: Mapped[User | None] = relationship("User", back_populates="audit_logs")

    __table_args__ = (Index("ix_audit_logs_action_created", "action", "created_at"),)
