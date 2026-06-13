"""Streamlit UI for saved SQLite records."""

from __future__ import annotations

import streamlit as st

from services.database_service import (
    clear_all_records,
    clear_saved_dataset_records,
    clear_saved_prediction_records,
    get_dataset_rows,
    get_dataset_uploads,
    get_prediction_history,
    get_record_counts,
)


@st.cache_data(show_spinner=False, ttl=10)
def get_cached_record_counts() -> dict[str, int]:
    """Cache lightweight SQLite summary counts for brief Home rerenders."""
    return get_record_counts()


@st.cache_data(show_spinner=False, ttl=10)
def get_cached_dataset_uploads():
    """Cache saved dataset upload listings for brief Home rerenders."""
    return get_dataset_uploads()


@st.cache_data(show_spinner=False, ttl=10)
def get_cached_dataset_rows(dataset_id: int):
    """Cache one saved dataset row preview for brief Home rerenders."""
    return get_dataset_rows(dataset_id)


@st.cache_data(show_spinner=False, ttl=10)
def get_cached_prediction_history():
    """Cache saved prediction history for brief Home rerenders."""
    return get_prediction_history()


def clear_saved_records_cache() -> None:
    """Clear cached SQLite dashboard data after a mutation."""
    get_cached_record_counts.clear()
    get_cached_dataset_uploads.clear()
    get_cached_dataset_rows.clear()
    get_cached_prediction_history.clear()


def render_clear_records_controls() -> None:
    """Render guarded controls for clearing persisted SQLite records."""
    with st.expander("Manage saved SQLite records", expanded=False):
        st.caption("Clear saved dataset uploads, prediction history, or both. This only affects SQLite records.")

        clear_mode = st.radio(
            "Choose what to clear",
            options=["Saved datasets", "Prediction history", "Everything"],
            horizontal=True,
            key="clear_records_mode",
        )

        confirmation_text = st.text_input(
            "Type CLEAR to confirm",
            key="clear_records_confirmation",
            help="This confirmation helps prevent accidental deletion of saved records.",
        )
        can_clear = confirmation_text.strip().upper() == "CLEAR"

        if st.button("Clear selected records", width="stretch", type="secondary"):
            if not can_clear:
                st.warning("Type CLEAR in the confirmation box before deleting saved records.")
                return

            try:
                if clear_mode == "Saved datasets":
                    result = clear_saved_dataset_records()
                    clear_saved_records_cache()
                    for key in ("last_saved_upload_signature", "last_saved_dataset_id"):
                        st.session_state.pop(key, None)
                    st.success(
                        f"Cleared {result['datasets_deleted']:,} saved dataset(s) and {result['rows_deleted']:,} stored row snapshot(s)."
                    )
                elif clear_mode == "Prediction history":
                    deleted_predictions = clear_saved_prediction_records()
                    clear_saved_records_cache()
                    st.session_state.pop("last_prediction_id", None)
                    st.success(f"Cleared {deleted_predictions:,} saved prediction record(s).")
                else:
                    result = clear_all_records()
                    clear_saved_records_cache()
                    for key in ("last_saved_upload_signature", "last_saved_dataset_id", "last_prediction_id"):
                        st.session_state.pop(key, None)
                    st.success(
                        "Cleared "
                        f"{result['datasets_deleted']:,} dataset(s), "
                        f"{result['rows_deleted']:,} row snapshot(s), and "
                        f"{result['predictions_deleted']:,} prediction record(s)."
                    )

                for key in ("clear_records_confirmation", "clear_records_mode"):
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            except Exception as error:
                st.error("The saved records could not be cleared right now.")
                st.caption(f"Technical detail: {error}")


def render_saved_records_dashboard() -> None:
    """Display saved datasets and prediction history from SQLite."""
    try:
        counts = get_cached_record_counts()
    except Exception as error:
        st.warning("Saved records are unavailable right now.")
        st.caption(f"Technical detail: {error}")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Saved Datasets", f"{counts['datasets']:,}")
    metric_columns[1].metric("Saved Dataset Rows", f"{counts['rows']:,}")
    metric_columns[2].metric("Saved Predictions", f"{counts['predictions']:,}")

    render_clear_records_controls()

    with st.expander("Saved Datasets", expanded=False):
        st.subheader("Saved Dataset Uploads")
        try:
            uploads = get_cached_dataset_uploads()
        except Exception as error:
            st.warning("Saved dataset uploads could not be loaded.")
            st.caption(f"Technical detail: {error}")
            uploads = None

        if uploads is None:
            return

        if uploads.empty:
            st.info("No dataset uploads have been saved yet.")
        else:
            st.dataframe(uploads, width="stretch", hide_index=True)
            selected_dataset_id = st.selectbox(
                "Preview saved rows for dataset ID",
                options=uploads["dataset_id"].tolist(),
            )
            try:
                saved_rows = get_cached_dataset_rows(int(selected_dataset_id))
                st.dataframe(saved_rows, width="stretch", hide_index=True)
            except Exception as error:
                st.warning("Saved dataset rows could not be loaded.")
                st.caption(f"Technical detail: {error}")

    with st.expander("Prediction History", expanded=False):
        st.subheader("Saved Prediction History")
        try:
            predictions = get_cached_prediction_history()
        except Exception as error:
            st.warning("Saved prediction history could not be loaded.")
            st.caption(f"Technical detail: {error}")
            return

        if predictions.empty:
            st.info("No predictions have been saved yet.")
        else:
            st.dataframe(predictions, width="stretch", hide_index=True)
