"""Report generation UI for the Student Performance Prediction Dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from preprocessing.exploration_analysis import get_categorical_columns, get_numeric_columns
from services.database_service import get_dataset_uploads, get_prediction_history, get_record_counts
from utilities.dataset_manager import get_classification_run, get_clustering_run, get_report_dataset_state, get_schema_mapping
from utilities.database_ui import render_clear_records_controls
from utilities.trust_ui import (
    build_feature_importance_overview,
    render_data_governance_indicators,
    render_dataset_source_banner,
    render_preprocessing_audit_card,
)


def get_active_report_dataset(fallback_data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Select the best available human-readable dataset for reporting."""
    report_data, dataset_name, _source_label = get_report_dataset_state()
    if report_data is not None:
        return report_data, dataset_name or "Uploaded dataset"
    return fallback_data, "Uploaded dataset"


def get_schema_mapping_status() -> pd.DataFrame:
    """Build a compact schema-mapping summary table for reports."""
    mapping = get_schema_mapping() or {}
    return pd.DataFrame(
        [
            {"Role": "Student ID", "Mapped Column": mapping.get("student_id_column") or "Not mapped"},
            {"Role": "Target", "Mapped Column": mapping.get("target_column") or "Derived from score columns"},
            {"Role": "Attendance", "Mapped Column": mapping.get("attendance_column") or "Not mapped"},
            {"Role": "Scores", "Mapped Column": ", ".join(mapping.get("score_columns", []) or []) or "Not mapped"},
            {"Role": "Risk Indicators", "Mapped Column": ", ".join(mapping.get("risk_indicator_columns", []) or []) or "Not mapped"},
        ]
    )


def numeric_column_mean(data: pd.DataFrame, column: str) -> float | None:
    """Return a numeric mean for a report column when available."""
    if column not in data.columns:
        return None
    value = pd.to_numeric(data[column], errors="coerce").mean()
    return float(value) if value == value else None


def build_data_quality_report(data: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level data quality report."""
    return pd.DataFrame(
        {
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values,
            "Missing Values": data.isna().sum().values,
            "Missing Percentage": (data.isna().mean().values * 100).round(2),
            "Unique Values": data.nunique(dropna=True).values,
        }
    )


def build_executive_summary(data: pd.DataFrame, dataset_name: str) -> str:
    """Build a downloadable Markdown project summary."""
    numeric_columns = get_numeric_columns(data)
    categorical_columns = get_categorical_columns(data)
    missing_values = int(data.isna().sum().sum())
    mapping = get_schema_mapping() or {}
    score_columns = mapping.get("score_columns", []) or []
    performance_columns = mapping.get("performance_columns", []) or []
    final_grade_column = performance_columns[0] if performance_columns else score_columns[0] if score_columns else None
    schema_lines = [
        f"- student_id_column: {mapping.get('student_id_column') or 'not mapped'}",
        f"- target_column: {mapping.get('target_column') or 'derived from score columns'}",
        f"- attendance_column: {mapping.get('attendance_column') or 'not mapped'}",
        f"- score_columns: {', '.join(score_columns) if score_columns else 'not mapped'}",
        f"- risk_indicator_columns: {', '.join(mapping.get('risk_indicator_columns', []) or []) if mapping.get('risk_indicator_columns') else 'not mapped'}",
    ]
    average_final_grade = numeric_column_mean(data, final_grade_column) if final_grade_column else None
    average_final_grade_text = "N/A" if average_final_grade is None else f"{average_final_grade:.2f}"
    pass_rate_text = "N/A"
    if final_grade_column:
        final_grades = pd.to_numeric(data[final_grade_column], errors="coerce").dropna()
        if not final_grades.empty:
            threshold = 10 if float(final_grades.max()) <= 20 else 60
            pass_rate_text = f"{(final_grades.ge(threshold).mean() * 100):.1f}%"

    return f"""# Student Performance Prediction Dashboard Report

## Dataset
- Active dataset: {dataset_name}
- Rows: {len(data):,}
- Columns: {len(data.columns):,}
- Numeric columns: {len(numeric_columns):,}
- Categorical columns: {len(categorical_columns):,}
- Missing values: {missing_values:,}

## Dataset Schema Mapping
{chr(10).join(schema_lines)}

## Dynamic Academic Statistics
- Average selected performance score: {average_final_grade_text}
- Pass rate from mapped performance threshold: {pass_rate_text}

## Implemented Dashboard Modules
- Dataset upload and preview
- CSV, Excel, and table-based PDF upload support
- Data preprocessing
- Data exploration and visual analytics
- Pass/Fail prediction using Logistic Regression
- Classification comparison using Logistic Regression, Decision Tree, and Random Forest
- K-Means clustering for student segmentation
- SQLite persistence for datasets and prediction history

## Academic Interpretation
The dashboard supports academic decision-making by helping users identify performance trends, predict Pass/Fail outcomes, and group students into support categories.

## Recommendation
Use the prediction and clustering outputs together to identify students who need early academic support, attendance monitoring, tutoring, or advisor follow-up.
"""


def render_downloads(data: pd.DataFrame, dataset_name: str, quality_report: pd.DataFrame) -> None:
    """Render report export buttons."""
    summary_markdown = build_executive_summary(data, dataset_name)

    col_one, col_two, col_three = st.columns(3)
    with col_one:
        st.download_button(
            "Download Executive Report",
            data=summary_markdown,
            file_name="student_performance_report.md",
            mime="text/markdown",
            width="stretch",
        )
    with col_two:
        st.download_button(
            "Download Data Quality CSV",
            data=quality_report.to_csv(index=False),
            file_name="data_quality_report.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_three:
        st.download_button(
            "Download Dataset Preview CSV",
            data=data.head(100).to_csv(index=False),
            file_name="dataset_preview.csv",
            mime="text/csv",
            width="stretch",
        )


def render_model_outputs_summary() -> None:
    """Display saved model outputs from session state when available."""
    run = get_classification_run()
    if run is not None:
        st.subheader("Latest Classification Output")
        metrics = [
            {
                "Model": result.model_name,
                "Accuracy": round(result.accuracy, 4),
                "Precision": round(result.precision, 4),
                "Recall": round(result.recall, 4),
                "F1-score": round(result.f1_score, 4),
                "ROC-AUC": round(result.roc_auc, 4) if result.roc_auc is not None else None,
                "PR-AUC": round(result.average_precision, 4) if result.average_precision is not None else None,
                "Brier Score": round(result.brier_score, 4) if result.brier_score is not None else None,
            }
            for result in run.results
        ]
        st.dataframe(pd.DataFrame(metrics), width="stretch", hide_index=True)
        if getattr(run, "performance_summary", None):
            st.caption(run.performance_summary)
        feature_overview = build_feature_importance_overview(run)
        if not feature_overview.empty:
            st.subheader("Feature Importance Overview")
            st.dataframe(feature_overview, width="stretch", hide_index=True)
    else:
        st.info("No classification run is available in this session yet.")

    run = get_clustering_run()
    if run is not None:
        st.subheader("Latest Clustering Output")
        st.dataframe(run.interpretations, width="stretch", hide_index=True)
    else:
        st.info("No clustering run is available in this session yet.")


def render_database_report() -> None:
    """Display database-level reporting tables."""
    try:
        counts = get_record_counts()
        uploads = get_dataset_uploads()
        predictions = get_prediction_history()
    except Exception as error:
        st.warning("Database report could not be loaded.")
        st.caption(f"Technical detail: {error}")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Saved Datasets", f"{counts['datasets']:,}")
    metric_columns[1].metric("Saved Rows", f"{counts['rows']:,}")
    metric_columns[2].metric("Saved Predictions", f"{counts['predictions']:,}")

    render_clear_records_controls()

    st.subheader("Saved Uploads")
    if uploads.empty:
        st.info("No saved uploads yet.")
    else:
        st.dataframe(uploads, width="stretch", hide_index=True)

    st.subheader("Prediction History")
    if predictions.empty:
        st.info("No saved predictions yet.")
    else:
        st.dataframe(predictions, width="stretch", hide_index=True)
        st.download_button(
            "Download Prediction History CSV",
            data=predictions.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )


def render_reports_dashboard(fallback_data: pd.DataFrame) -> None:
    """Render the complete reports page."""
    data, dataset_name = get_active_report_dataset(fallback_data)
    quality_report = build_data_quality_report(data)
    numeric_columns = get_numeric_columns(data)
    categorical_columns = get_categorical_columns(data)
    missing_values = int(data.isna().sum().sum())
    schema_status = get_schema_mapping_status()
    mapped_roles = int((schema_status["Mapped Column"] != "Not mapped").sum())

    st.markdown(
        f"""
        <div class="module-intro">
            <strong>Active report dataset: {dataset_name}</strong>
            <span>Generate summaries, export report files, and review saved dashboard records.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_dataset_source_banner()
    render_data_governance_indicators()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(data):,}")
    metric_columns[1].metric("Columns", f"{len(data.columns):,}")
    metric_columns[2].metric("Numeric Fields", f"{len(numeric_columns):,}")
    metric_columns[3].metric("Missing Values", f"{missing_values:,}")

    st.metric("Mapped Schema Roles", f"{mapped_roles}/{len(schema_status)}")

    overview_tab, quality_tab, outputs_tab, database_tab, export_tab = st.tabs(
        ["Overview", "Data Quality", "Model Outputs", "Saved Records", "Exports"]
    )

    with overview_tab:
        st.subheader("Dataset Preview")
        st.dataframe(data.head(50), width="stretch")
        st.subheader("Schema Mapping")
        st.dataframe(schema_status, width="stretch", hide_index=True)
        st.subheader("Field Summary")
        st.write(f"Numeric fields: {', '.join(numeric_columns) if numeric_columns else 'None'}")
        st.write(f"Categorical fields: {', '.join(categorical_columns) if categorical_columns else 'None'}")

    with quality_tab:
        st.subheader("Data Quality Report")
        st.dataframe(quality_report, width="stretch", hide_index=True)
        render_preprocessing_audit_card()

    with outputs_tab:
        render_model_outputs_summary()

    with database_tab:
        render_database_report()

    with export_tab:
        st.subheader("Export Reports")
        render_downloads(data, dataset_name, quality_report)
