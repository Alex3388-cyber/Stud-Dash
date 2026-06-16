"""Lightweight Streamlit entry point for the Student Performance Prediction Dashboard."""

from __future__ import annotations

import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, StopException

from services.database_service import initialize_storage, get_latest_dataset, record_audit_event
from services.dataset_service import ensure_active_dataset_pipeline
from ui.pages import PAGE_RENDERERS
from ui.shell import load_custom_css, render_sidebar
from utilities.constants import APP_TITLE
from utilities.dataset_manager import (
    build_dataset_signature,
    has_active_dataset,
    migrate_legacy_session_state,
    set_raw_dataset,
    set_schema_mapping,
)
from utilities.schema_mapping import merge_schema_mapping


def _restore_dataset_from_sqlite() -> bool:
    """Load the most recent saved dataset from SQLite when the session is empty.

    Browser navigations start a new WebSocket session and wipe session state.
    This restores the dataset silently so every page can access it without
    the user having to re-upload.
    """
    if has_active_dataset():
        return False
    try:
        data, file_name, _dataset_id = get_latest_dataset()
    except Exception:
        return False
    if data is None or data.empty:
        return False

    dataset_name = file_name or "Restored dataset"
    sig = build_dataset_signature(dataset_name, data)
    set_raw_dataset(data, dataset_name=dataset_name, source="sqlite", signature=sig)
    detected_mapping = merge_schema_mapping(data, None)
    set_schema_mapping(detected_mapping, sig)
    try:
        record_audit_event("dataset_restore", entity_name=dataset_name, detail=f"{len(data):,} rows restored from SQLite", rows_affected=len(data))
    except Exception:
        pass
    return True


def main() -> None:
    """Initialize shared services and render the selected page."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        initialize_storage()
    except Exception:
        st.warning("Database setup could not be completed. Saved records may be unavailable.")

    try:
        from database.pg_connection import init_db
        init_db()
    except Exception:
        pass

    migrate_legacy_session_state()
    restored = _restore_dataset_from_sqlite()
    if restored:
        st.toast("Dataset restored from saved records.", icon="✅")

    # show_spinner=True enables the step-by-step progress bar the first time
    # the pipeline runs for a given dataset (skipped on subsequent navigations).
    ensure_active_dataset_pipeline(show_spinner=True)
    load_custom_css()

    selected_page = render_sidebar()
    try:
        PAGE_RENDERERS[selected_page]()
    except (RerunException, StopException):
        raise
    except Exception:
        st.error("Something went wrong while loading this page.")
        st.info("Try refreshing the page or returning to Home. If the issue continues, check the uploaded dataset format.")


if __name__ == "__main__":
    main()
