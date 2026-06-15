"""Lightweight Streamlit entry point for the Student Performance Prediction Dashboard."""

from __future__ import annotations

import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, StopException

from services.database_service import initialize_storage
from services.dataset_service import ensure_active_dataset_pipeline
from ui.pages import PAGE_RENDERERS
from ui.shell import load_custom_css, render_sidebar
from utilities.constants import APP_TITLE
from utilities.dataset_manager import migrate_legacy_session_state


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
    ensure_active_dataset_pipeline(show_spinner=False)
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
