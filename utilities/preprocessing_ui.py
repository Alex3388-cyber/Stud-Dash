"""Streamlit UI for dataset preprocessing.

This module presents preprocessing results without training any model. The
actual transformation logic lives in preprocessing/preprocessor.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.preprocessing_service import build_summary_table, run_preprocessing
from utilities.dataset_manager import (
    get_dataset_signature,
    get_feature_matrix,
    get_feature_matrix_signature,
    get_preprocessing_summary,
    get_cleaned_dataset,
    set_preprocessing_artifacts,
)


def render_column_list(title: str, columns: list[str]) -> None:
    """Display detected columns in a compact, readable format."""
    st.subheader(title)
    if columns:
        st.write(", ".join(columns))
    else:
        st.caption("No columns detected for this step.")


def render_warning_list(messages: list[str]) -> None:
    """Display preprocessing warnings in a user-friendly way."""
    if not messages:
        st.success("No identifier or high-cardinality warnings were detected.")
        return

    for message in messages:
        st.warning(message)


def render_preprocessing_results(cleaned_data: pd.DataFrame, feature_matrix: pd.DataFrame, summary) -> None:
    """Render the persisted preprocessing outputs for the active dataset."""
    st.success("Preprocessing completed successfully.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows Processed", f"{summary.processed_rows:,}")
    metric_columns[1].metric("Duplicates Removed", f"{summary.duplicate_rows_removed:,}")
    metric_columns[2].metric("Missing Before", f"{summary.missing_values_before:,}")
    metric_columns[3].metric("Matrix Format", summary.feature_matrix_format.title())

    tabs = st.tabs(
        ["Cleaned Preview", "Feature Matrix Preview", "Preprocessing Summary", "Detected Columns", "Warnings"]
    )

    with tabs[0]:
        st.subheader("Cleaned Dataset Preview")
        st.caption("This view stays human-readable so reports and manual review use real academic values.")
        st.dataframe(cleaned_data.head(100), width="stretch")

    with tabs[1]:
        st.subheader("Feature Matrix Preview")
        st.caption("This transformed matrix is model-ready and includes encoded/scaled values.")
        st.dataframe(feature_matrix.head(100), width="stretch")

    with tabs[2]:
        st.subheader("Preprocessing Summary")
        st.dataframe(build_summary_table(summary), width="stretch", hide_index=True)

    with tabs[3]:
        left, middle, right = st.columns(3)
        with left:
            render_column_list("Normalized Numerical Features", summary.numeric_columns)
        with middle:
            render_column_list("Encoded Categorical Features", summary.categorical_columns)
        with right:
            render_column_list("Identifier / Excluded Columns", summary.identifier_columns + summary.high_cardinality_columns)

    with tabs[4]:
        st.subheader("Preprocessing Warnings")
        st.caption(
            "These checks protect the dashboard from meaningless identifier features and one-hot encoding blow-ups on very high-cardinality columns."
        )
        render_warning_list(summary.warning_messages)
        left, right = st.columns(2)
        with left:
            render_column_list("Identifier Columns Excluded", summary.identifier_columns)
        with right:
            render_column_list(
                f"High-Cardinality Columns Excluded (>{summary.cardinality_threshold} unique values)",
                summary.high_cardinality_columns,
            )


def render_preprocessing_module(data: pd.DataFrame | None) -> None:
    """Render preprocessing controls, transformed preview, and step summary."""
    st.markdown(
        """
        <div class="module-intro">
            <strong>Preprocessing Workspace</strong>
            <span>Clean the uploaded dataset before future model training or clustering.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data is None:
        st.info("Upload a dataset first to enable preprocessing.")
        return

    if data.empty:
        st.warning("The uploaded dataset is empty, so preprocessing cannot run.")
        return

    if len(data.columns) == 0:
        st.error("The uploaded dataset has no columns to preprocess.")
        return

    st.caption(
        "This step removes duplicates, imputes missing values, excludes identifier-like and high-cardinality fields from encoding, "
        "one-hot encodes safe categorical fields, and normalizes numerical fields with Scikit-learn."
    )

    dataset_signature = get_dataset_signature()
    stored_signature = get_feature_matrix_signature()
    stored_feature_matrix = get_feature_matrix()
    stored_cleaned_data = get_cleaned_dataset()
    stored_summary = get_preprocessing_summary()

    cardinality_threshold = st.slider(
        "Maximum unique categories for one-hot encoding",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Categorical columns above this threshold are excluded from one-hot encoding to keep the feature matrix scalable.",
    )

    if st.button("Run Preprocessing", type="primary", width="stretch"):
        try:
            with st.spinner("Cleaning and transforming the uploaded dataset..."):
                cleaned_data, feature_matrix, summary, _transformer = run_preprocessing(
                    data,
                    cardinality_threshold=int(cardinality_threshold),
                )
        except Exception as error:
            st.error(f"Preprocessing failed: {error}")
            return

        # Keep the cleaned dataset and model-ready matrix available separately
        # so reports/charts use readable values and models use transformed ones.
        set_preprocessing_artifacts(cleaned_data, feature_matrix, summary)

        stored_cleaned_data = cleaned_data
        stored_feature_matrix = feature_matrix
        stored_summary = summary
        stored_signature = dataset_signature

    if (
        stored_cleaned_data is not None
        and stored_feature_matrix is not None
        and stored_summary is not None
        and stored_signature == dataset_signature
    ):
        render_preprocessing_results(stored_cleaned_data, stored_feature_matrix, stored_summary)
        return

    st.info("Click the button to generate a cleaned dataset preview and a separate transformed feature matrix.")
