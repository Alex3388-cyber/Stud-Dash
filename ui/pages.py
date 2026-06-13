"""Page-level UI orchestration for the Streamlit dashboard."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from services.dataset_service import (
    ensure_active_dataset_pipeline,
    get_active_kpi_dataset,
    get_active_modeling_dataset,
    get_analysis_dataset,
    get_exploration_data_sources,
)
from ui.home_page import render_ai_insights_panel, render_home
from ui.shell import render_page_header, render_panel
from utilities.classification_ui import render_classification_module
from utilities.clustering_ui import render_kmeans_clustering_module
from utilities.data_exploration import render_data_exploration_module
from utilities.dataset_manager import get_dataset_name, get_dataset_origin, get_raw_dataset
from utilities.dataset_upload import render_dataset_upload_module
from utilities.prediction_form import render_student_prediction_form
from utilities.preprocessing_ui import render_preprocessing_module
from utilities.reports_ui import render_reports_dashboard


def render_upload_dataset() -> None:
    """Render the dataset upload page with dataset inspection and preprocessing."""
    render_page_header(
        "Upload Dataset",
        "Upload a CSV or Excel file, preview records, and review key dataset quality indicators.",
        "Data Intake",
    )

    left, right = st.columns([1.35, 0.85])
    with left:
        uploaded_data = render_dataset_upload_module()

    with right:
        render_panel(
            "Upload Checklist",
            "Use a structured dataset so later exploration and modeling steps remain reliable.",
            [
                "Use one row per student or assessment record",
                "Keep column names clear and consistent",
                "Include performance or assessment fields",
                "Review missing values before analysis",
                "Prefer CSV or XLSX for best compatibility",
            ],
        )

    active_data = uploaded_data if uploaded_data is not None else get_raw_dataset()
    pipeline_ran, pipeline_report = ensure_active_dataset_pipeline(show_spinner=uploaded_data is not None)
    if isinstance(active_data, pd.DataFrame) and not active_data.empty:
        active_name = get_dataset_name()
        active_source = "ready for dashboard" if get_dataset_origin() == "upload" else get_dataset_origin()
        st.markdown(
            f"""
            <div class="status-banner">
                <span class="status-dot"></span>
                <strong>Dashboard sync active</strong>
                <span>{escape(str(active_name))} now powers Home, Exploration, Prediction, Clustering, and Reports | {active_source}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if pipeline_ran and pipeline_report:
            st.success(
                "Dataset synced across the dashboard. "
                f"Preprocessing: {pipeline_report['preprocessing']} | "
                f"Classification: {pipeline_report['classification']} | "
                f"Clustering: {pipeline_report['clustering']}"
            )

    render_preprocessing_module(active_data)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        render_panel(
            "What This Module Shows",
            "The upload page now performs immediate Pandas-based inspection.",
            [
                "Row and column counts",
                "Column names and data types",
                "Missing-value counts and percentages",
                "Interactive preview and summary statistics",
            ],
        )
    with bottom_right:
        render_panel(
            "Data Pipeline Status",
            "Upload validation, SQLite persistence, and preprocessing are now active.",
            [
                "Invalid uploads are handled with friendly messages",
                "Uploaded datasets are saved to SQLite",
                "Preprocessed data is ready for exploration and modeling",
            ],
        )


def render_data_exploration() -> None:
    """Render the data exploration page with Plotly analysis."""
    ensure_active_dataset_pipeline(show_spinner=False)
    render_page_header(
        "Data Exploration",
        "Explore summary statistics, frequency patterns, correlations, and interactive Plotly charts.",
        "Academic Review",
    )

    data_sources = get_exploration_data_sources()
    if not data_sources:
        st.info("Upload a student dataset from the Home page to explore real values.")
        return

    default_source = "Cleaned dataset" if "Cleaned dataset" in data_sources else "Uploaded dataset"
    source_options = list(data_sources.keys())
    selected_source = st.radio(
        "Dataset source",
        source_options,
        horizontal=True,
        index=source_options.index(default_source),
        key="exploration_dataset_source",
    )
    data, dataset_name = data_sources[selected_source]
    render_data_exploration_module(data, dataset_name)


def render_insights() -> None:
    """Render the dedicated AI academic insights page."""
    ensure_active_dataset_pipeline(show_spinner=False)
    render_page_header(
        "AI Academic Insights",
        "Review dynamic academic alerts, performance patterns, risk signals, and intervention recommendations.",
        "Insights",
    )

    data, dataset_name, source_label = get_active_kpi_dataset()
    if data.empty:
        st.info("Upload a student dataset from the Home page to generate AI insights.")
        return

    st.markdown(
        f"""
        <div class="status-banner">
            <span class="status-dot"></span>
            <strong>Insights ready</strong>
            <span>{escape(str(dataset_name))} | {len(data):,} record(s) | {escape(str(source_label))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    from services.analytics_service import build_home_kpis

    snapshot = build_home_kpis()
    render_ai_insights_panel(snapshot)


def render_prediction() -> None:
    """Render the live prediction console and classification tools."""
    ensure_active_dataset_pipeline(show_spinner=False)
    render_page_header(
        "Student Performance Prediction",
        "Predict Pass or Fail from study time, absences, failures, and previous grades.",
        "Prediction",
    )

    active_data, dataset_name = get_active_modeling_dataset()
    if active_data is None or dataset_name is None:
        st.info("Upload a student dataset from the Home page before using prediction and classification.")
        return

    form_tab, training_tab = st.tabs(["Prediction Form", "Model Training and Comparison"])
    with form_tab:
        render_student_prediction_form(active_data, dataset_name)
    with training_tab:
        render_classification_module(active_data, dataset_name)


def render_clustering() -> None:
    """Render the K-Means clustering page."""
    ensure_active_dataset_pipeline(show_spinner=False)
    render_page_header(
        "K-Means Clustering",
        "Segment students into high performers, average performers, and at-risk students.",
        "Student Segmentation",
    )

    active_data, dataset_name = get_active_modeling_dataset()
    if active_data is None or dataset_name is None:
        st.info("Upload a student dataset from the Home page before running K-Means clustering.")
        return

    render_kmeans_clustering_module(active_data, dataset_name)


def render_reports() -> None:
    """Render generated reports and export options."""
    ensure_active_dataset_pipeline(show_spinner=False)
    render_page_header(
        "Reports",
        "Prepare academic summary outputs for leadership, advisors, instructors, and student support teams.",
        "Academic Reporting",
    )

    data, _dataset_name = get_analysis_dataset()
    if data is None:
        st.info("Upload a student dataset from the Home page before generating reports.")
        return

    render_reports_dashboard(data)


PAGE_RENDERERS = {
    "Home": render_home,
    "Insights": render_insights,
    "Data Exploration": render_data_exploration,
    "Prediction": render_prediction,
    "Clustering": render_clustering,
    "Reports": render_reports,
}
