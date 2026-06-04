"""Centralized dataset lifecycle management for Streamlit session state.

This module keeps raw uploads, cleaned human-readable datasets, transformed
feature matrices, and report-safe datasets separate. It reduces fragile
cross-module session state usage and makes each dashboard page explicit about
which dataset stage it needs.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import streamlit as st


RAW_DATASET_KEY = "raw_dataset"
CLEANED_DATASET_KEY = "cleaned_dataset"
FEATURE_MATRIX_KEY = "feature_matrix"
REPORT_DATASET_KEY = "report_dataset"
DATASET_NAME_KEY = "dataset_name"
DATASET_SIGNATURE_KEY = "dataset_signature"
DATASET_ORIGIN_KEY = "dataset_origin"
FEATURE_MATRIX_SIGNATURE_KEY = "feature_matrix_signature"
PREPROCESSING_SUMMARY_KEY = "preprocessing_summary"
CLASSIFICATION_RUN_KEY = "classification_run"
CLUSTERING_RUN_KEY = "clustering_run"
AUTO_PIPELINE_SIGNATURE_KEY = "auto_pipeline_signature"
AUTO_PIPELINE_REPORT_KEY = "auto_pipeline_report"
SCHEMA_MAPPING_KEY = "dataset_schema_mapping"
SCHEMA_MAPPING_SIGNATURE_KEY = "dataset_schema_signature"


LEGACY_DATASET_KEYS = (
    "uploaded_dataset",
    "uploaded_dataset_name",
    "active_dataset_signature",
    "active_dataset_restored",
    "preprocessed_dataset",
    "preprocessed_dataset_signature",
)

DERIVED_STATE_KEYS = (
    CLEANED_DATASET_KEY,
    FEATURE_MATRIX_KEY,
    REPORT_DATASET_KEY,
    FEATURE_MATRIX_SIGNATURE_KEY,
    PREPROCESSING_SUMMARY_KEY,
    SCHEMA_MAPPING_KEY,
    SCHEMA_MAPPING_SIGNATURE_KEY,
    CLASSIFICATION_RUN_KEY,
    CLUSTERING_RUN_KEY,
    AUTO_PIPELINE_SIGNATURE_KEY,
    AUTO_PIPELINE_REPORT_KEY,
    "last_prediction_id",
)

ANALYTICS_STATE_KEYS = (
    CLASSIFICATION_RUN_KEY,
    CLUSTERING_RUN_KEY,
    AUTO_PIPELINE_SIGNATURE_KEY,
    AUTO_PIPELINE_REPORT_KEY,
    "last_prediction_id",
)


def _copy_dataframe(data: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a defensive dataframe copy for session storage."""
    if not isinstance(data, pd.DataFrame):
        return None
    return data.copy(deep=True)


def build_dataset_signature(dataset_name: str, data: pd.DataFrame) -> tuple[str, int, tuple[str, ...]]:
    """Create a stable signature for one uploaded dataset."""
    return dataset_name, len(data), tuple(map(str, data.columns))


def migrate_legacy_session_state() -> None:
    """Move older dataset keys into the centralized dataset structure once."""
    session = st.session_state
    if RAW_DATASET_KEY in session:
        return

    legacy_raw = session.get("uploaded_dataset")
    if isinstance(legacy_raw, pd.DataFrame):
        dataset_name = str(session.get("uploaded_dataset_name", "Uploaded dataset"))
        signature = session.get("active_dataset_signature")
        set_raw_dataset(
            legacy_raw,
            dataset_name=dataset_name,
            source="upload",
            signature=signature if signature is not None else build_dataset_signature(dataset_name, legacy_raw),
            reset_derived=False,
        )

    legacy_feature_matrix = session.get("preprocessed_dataset")
    if isinstance(legacy_feature_matrix, pd.DataFrame):
        session[FEATURE_MATRIX_KEY] = _copy_dataframe(legacy_feature_matrix)
        session[FEATURE_MATRIX_SIGNATURE_KEY] = session.get(
            "preprocessed_dataset_signature",
            session.get(DATASET_SIGNATURE_KEY),
        )

    for key in LEGACY_DATASET_KEYS:
        session.pop(key, None)


def clear_derived_state(widget_keys: Iterable[str] | None = None) -> None:
    """Remove datasets and analytics artifacts derived from the raw upload."""
    session = st.session_state
    for key in DERIVED_STATE_KEYS:
        session.pop(key, None)

    if widget_keys:
        for key in widget_keys:
            session.pop(key, None)


def clear_analytics_state(widget_keys: Iterable[str] | None = None) -> None:
    """Remove analytics artifacts that depend on the current schema mapping."""
    session = st.session_state
    for key in ANALYTICS_STATE_KEYS:
        session.pop(key, None)

    if widget_keys:
        for key in widget_keys:
            session.pop(key, None)


def clear_active_dataset(widget_keys: Iterable[str] | None = None) -> None:
    """Remove the active dataset and every dataset-derived artifact."""
    clear_derived_state(widget_keys=widget_keys)
    session = st.session_state
    for key in (RAW_DATASET_KEY, DATASET_NAME_KEY, DATASET_SIGNATURE_KEY, DATASET_ORIGIN_KEY):
        session.pop(key, None)


def set_raw_dataset(
    data: pd.DataFrame,
    dataset_name: str,
    source: str = "upload",
    signature: tuple[str, int, tuple[str, ...]] | None = None,
    reset_derived: bool = True,
    widget_keys: Iterable[str] | None = None,
) -> None:
    """Register a new raw dataset as the active dashboard dataset."""
    if reset_derived:
        clear_derived_state(widget_keys=widget_keys)

    session = st.session_state
    session[RAW_DATASET_KEY] = _copy_dataframe(data)
    session[REPORT_DATASET_KEY] = _copy_dataframe(data)
    session[DATASET_NAME_KEY] = dataset_name
    session[DATASET_SIGNATURE_KEY] = signature if signature is not None else build_dataset_signature(dataset_name, data)
    session[DATASET_ORIGIN_KEY] = source


def set_preprocessing_artifacts(
    cleaned_dataset: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    preprocessing_summary: Any,
) -> None:
    """Store cleaned and transformed datasets produced by preprocessing."""
    session = st.session_state
    session[CLEANED_DATASET_KEY] = _copy_dataframe(cleaned_dataset)
    session[FEATURE_MATRIX_KEY] = _copy_dataframe(feature_matrix)
    session[REPORT_DATASET_KEY] = _copy_dataframe(cleaned_dataset)
    session[PREPROCESSING_SUMMARY_KEY] = preprocessing_summary
    session[FEATURE_MATRIX_SIGNATURE_KEY] = session.get(DATASET_SIGNATURE_KEY)


def set_classification_run(run: Any) -> None:
    """Persist the latest classification run for reuse across pages."""
    st.session_state[CLASSIFICATION_RUN_KEY] = run


def get_classification_run() -> Any:
    """Return the latest classification run."""
    return st.session_state.get(CLASSIFICATION_RUN_KEY)


def clear_classification_run() -> None:
    """Remove the latest classification run from session state."""
    st.session_state.pop(CLASSIFICATION_RUN_KEY, None)


def set_clustering_run(run: Any) -> None:
    """Persist the latest clustering run for reuse across pages."""
    st.session_state[CLUSTERING_RUN_KEY] = run


def get_clustering_run() -> Any:
    """Return the latest clustering run."""
    return st.session_state.get(CLUSTERING_RUN_KEY)


def clear_clustering_run() -> None:
    """Remove the latest clustering run from session state."""
    st.session_state.pop(CLUSTERING_RUN_KEY, None)


def set_auto_pipeline_report(signature: Any, report: Any) -> None:
    """Persist the latest automatic pipeline signature and report."""
    st.session_state[AUTO_PIPELINE_SIGNATURE_KEY] = signature
    st.session_state[AUTO_PIPELINE_REPORT_KEY] = report


def get_auto_pipeline_signature() -> Any:
    """Return the signature for the last automatic dataset pipeline run."""
    return st.session_state.get(AUTO_PIPELINE_SIGNATURE_KEY)


def get_auto_pipeline_report() -> Any:
    """Return the latest automatic dataset pipeline report."""
    return st.session_state.get(AUTO_PIPELINE_REPORT_KEY)


def set_schema_mapping(mapping: dict[str, Any], signature: Any) -> None:
    """Persist the active dataset schema mapping for the current dataset."""
    st.session_state[SCHEMA_MAPPING_KEY] = dict(mapping)
    st.session_state[SCHEMA_MAPPING_SIGNATURE_KEY] = signature


def get_schema_mapping() -> dict[str, Any] | None:
    """Return the stored schema mapping for the active session."""
    mapping = st.session_state.get(SCHEMA_MAPPING_KEY)
    return dict(mapping) if isinstance(mapping, dict) else None


def get_schema_mapping_signature() -> Any:
    """Return the dataset signature tied to the current schema mapping."""
    return st.session_state.get(SCHEMA_MAPPING_SIGNATURE_KEY)


def clear_schema_mapping() -> None:
    """Remove the current dataset schema mapping."""
    for key in (SCHEMA_MAPPING_KEY, SCHEMA_MAPPING_SIGNATURE_KEY):
        st.session_state.pop(key, None)


def get_dataset_name(default: str = "Uploaded dataset") -> str:
    """Return the active dataset name."""
    return str(st.session_state.get(DATASET_NAME_KEY, default))


def get_dataset_signature() -> Any:
    """Return the active dataset signature."""
    return st.session_state.get(DATASET_SIGNATURE_KEY)


def get_dataset_origin(default: str = "upload") -> str:
    """Return the current dataset origin label."""
    return str(st.session_state.get(DATASET_ORIGIN_KEY, default))


def get_raw_dataset() -> pd.DataFrame | None:
    """Return the original uploaded dataset."""
    data = st.session_state.get(RAW_DATASET_KEY)
    return data if isinstance(data, pd.DataFrame) else None


def get_cleaned_dataset() -> pd.DataFrame | None:
    """Return the cleaned human-readable dataset when available."""
    data = st.session_state.get(CLEANED_DATASET_KEY)
    return data if isinstance(data, pd.DataFrame) else None


def get_feature_matrix() -> pd.DataFrame | None:
    """Return the transformed model-ready feature matrix when available."""
    data = st.session_state.get(FEATURE_MATRIX_KEY)
    return data if isinstance(data, pd.DataFrame) else None


def get_feature_matrix_signature() -> Any:
    """Return the source dataset signature for the current feature matrix."""
    return st.session_state.get(FEATURE_MATRIX_SIGNATURE_KEY)


def get_report_dataset() -> pd.DataFrame | None:
    """Return the human-readable dataset used by reports and charts."""
    data = st.session_state.get(REPORT_DATASET_KEY)
    return data if isinstance(data, pd.DataFrame) else None


def get_preprocessing_summary() -> Any:
    """Return the latest preprocessing summary."""
    return st.session_state.get(PREPROCESSING_SUMMARY_KEY)


def get_dashboard_dataset() -> tuple[pd.DataFrame | None, str | None, str]:
    """Return the best human-readable dataset for KPI and chart rendering."""
    cleaned_dataset = get_cleaned_dataset()
    if cleaned_dataset is not None and not cleaned_dataset.empty:
        return cleaned_dataset, get_dataset_name(), "Cleaned dataset"

    report_dataset = get_report_dataset()
    if report_dataset is not None and not report_dataset.empty:
        return report_dataset, get_dataset_name(), "Uploaded dataset"

    raw_dataset = get_raw_dataset()
    if raw_dataset is not None and not raw_dataset.empty:
        return raw_dataset, get_dataset_name(), "Uploaded dataset"

    return None, None, "Upload required"


def get_modeling_dataset() -> tuple[pd.DataFrame | None, str | None, str]:
    """Return the best dataset for modeling and prediction workflows."""
    cleaned_dataset = get_cleaned_dataset()
    if cleaned_dataset is not None and not cleaned_dataset.empty:
        return cleaned_dataset, get_dataset_name(), "Cleaned dataset"

    raw_dataset = get_raw_dataset()
    if raw_dataset is not None and not raw_dataset.empty:
        return raw_dataset, get_dataset_name(), "Uploaded dataset"

    return None, None, "Upload required"


def get_report_dataset_state() -> tuple[pd.DataFrame | None, str | None, str]:
    """Return the dataset that reports should use.

    Reports always prefer human-readable datasets and never the transformed
    feature matrix.
    """
    report_dataset = get_report_dataset()
    if report_dataset is not None and not report_dataset.empty:
        return report_dataset, get_dataset_name(), "Report-ready dataset"

    raw_dataset = get_raw_dataset()
    if raw_dataset is not None and not raw_dataset.empty:
        return raw_dataset, get_dataset_name(), "Uploaded dataset"

    return None, None, "Upload required"


def has_active_dataset() -> bool:
    """Return True when a raw dataset is active in the current session."""
    data = get_raw_dataset()
    return isinstance(data, pd.DataFrame) and not data.empty
