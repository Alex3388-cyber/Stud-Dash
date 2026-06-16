"""Streamlit dataset upload module for the dashboard.

The module reads CSV or Excel files with Pandas, previews the uploaded dataset,
and displays essential data quality information for academic datasets.
"""

from __future__ import annotations

import hashlib
from io import StringIO
from typing import Any

import fitz
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import streamlit as st

from preprocessing.dataset_inspection import DatasetProfile, build_dataset_profile
from services.database_service import save_dataset_upload, record_audit_event, record_etl_job
from utilities.dataset_manager import (
    clear_analytics_state,
    clear_derived_state,
    get_dataset_name,
    get_raw_dataset,
    get_dataset_signature,
    get_schema_mapping,
    get_schema_mapping_signature,
    set_schema_mapping,
    set_raw_dataset,
)
from utilities.schema_mapping import (
    build_schema_status_table,
    mapping_supports_classification,
    mapping_supports_clustering,
    mapping_supports_prediction,
    merge_schema_mapping,
)
from utilities.trust_ui import render_data_governance_indicators
from utilities.validation import (
    display_validation_errors,
    validate_dataframe_for_dashboard,
    validate_uploaded_file,
)


SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "pdf"]
CSV_EXCEL_FILE_TYPES = ["csv", "xlsx", "xls"]
DATASET_DEPENDENT_WIDGET_KEYS = [
    "exploration_dataset_source",
    "summary_columns",
    "frequency_column",
    "correlation_columns",
    "bar_x",
    "bar_y",
    "histogram_x",
    "histogram_color",
    "scatter_x",
    "scatter_y",
    "scatter_color",
    "pie_column",
    "pie_top_n",
    "line_x",
    "line_y",
]


def build_uploaded_file_signature(uploaded_file: Any, data: pd.DataFrame) -> tuple[str, int, tuple[str, ...], str]:
    """Create a robust signature so new uploads always refresh dashboard state."""
    try:
        file_bytes = uploaded_file.getvalue()
    except Exception:
        file_bytes = b""

    fingerprint = hashlib.sha1(file_bytes).hexdigest()[:12] if file_bytes else "no-bytes"
    return str(uploaded_file.name), len(data), tuple(map(str, data.columns)), fingerprint


def make_unique_columns(columns: list[str]) -> list[str]:
    """Create safe, non-empty, unique column names from extracted PDF headers."""
    seen: dict[str, int] = {}
    unique_columns: list[str] = []

    for index, column in enumerate(columns, start=1):
        clean_column = str(column).strip() if column is not None else ""
        if not clean_column:
            clean_column = f"column_{index}"

        count = seen.get(clean_column, 0)
        seen[clean_column] = count + 1
        unique_columns.append(clean_column if count == 0 else f"{clean_column}_{count + 1}")

    return unique_columns


def table_rows_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame | None:
    """Convert extracted PDF table rows into a dataframe."""
    cleaned_rows = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in rows
    ]
    cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]

    if len(cleaned_rows) < 2:
        return None

    max_width = max(len(row) for row in cleaned_rows)
    normalized_rows = [row + [""] * (max_width - len(row)) for row in cleaned_rows]
    columns = make_unique_columns(normalized_rows[0])
    data = pd.DataFrame(normalized_rows[1:], columns=columns)
    return data.replace("", pd.NA)


def parse_delimited_pdf_text(text: str) -> pd.DataFrame | None:
    """Parse PDF text that contains CSV-like, tabular lines."""
    for separator in ["|", "\t", ",", ";"]:
        lines = [line.strip() for line in text.splitlines() if separator in line]
        if len(lines) < 2:
            continue

        try:
            parsed = pd.read_csv(StringIO("\n".join(lines)), sep=separator)
        except Exception:
            continue

        if not parsed.empty and len(parsed.columns) > 1:
            parsed.columns = make_unique_columns([str(column) for column in parsed.columns])
            return parsed

    return None


def read_pdf_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Extract a tabular dataset from a PDF file.

    PDF support is intended for PDFs that contain actual tables or clear
    delimited text. Scanned/image-only PDFs need OCR before they can become a
    usable dataset.
    """
    pdf_bytes = uploaded_file.read()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_tables: list[pd.DataFrame] = []
    extracted_text: list[str] = []

    for page in document:
        extracted_text.append(page.get_text())

        # PyMuPDF can detect table structures in many digitally generated PDFs.
        if hasattr(page, "find_tables"):
            tables = page.find_tables()
            for table in tables.tables:
                dataframe = table_rows_to_dataframe(table.extract())
                if dataframe is not None and not dataframe.empty:
                    extracted_tables.append(dataframe)

    if extracted_tables:
        return pd.concat(extracted_tables, ignore_index=True, sort=False)

    parsed_text_table = parse_delimited_pdf_text("\n".join(extracted_text))
    if parsed_text_table is not None:
        return parsed_text_table

    raise ValueError(
        "No usable table was found in the PDF. Upload a PDF with a selectable table, or convert the dataset to CSV/Excel."
    )


def read_uploaded_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded CSV, Excel, or PDF file into a Pandas dataframe."""
    file_name = uploaded_file.name.lower()

    # CSV files are read directly from Streamlit's uploaded file buffer.
    if file_name.endswith(".csv"):
        # The original UCI Student Performance CSV files are semicolon-delimited.
        # sep=None lets Pandas sniff commas, semicolons, tabs, and similar delimiters.
        return pd.read_csv(uploaded_file, sep=None, engine="python")

    # Excel files use Pandas' Excel reader. The dependency is listed in requirements.txt.
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    # PDF files are supported when they contain extractable table data.
    if file_name.endswith(".pdf"):
        return read_pdf_dataset(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV, Excel, or PDF file.")


def render_overview_metrics(profile: DatasetProfile) -> None:
    """Display row, column, missing-value, and completeness metrics."""
    total_missing = int(profile.missing_values["Missing Values"].sum())
    total_cells = profile.row_count * profile.column_count
    completeness = 100 if total_cells == 0 else 100 - ((total_missing / total_cells) * 100)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{profile.row_count:,}")
    metric_columns[1].metric("Columns", f"{profile.column_count:,}")
    metric_columns[2].metric("Missing Values", f"{total_missing:,}")
    metric_columns[3].metric("Completeness", f"{completeness:.1f}%")


def render_schema_detection_wizard(data: pd.DataFrame, dataset_signature: tuple[str, int, tuple[str, ...]]) -> dict[str, Any]:
    """Render a schema mapping wizard so custom academic datasets can be aligned."""
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    all_columns = list(map(str, data.columns))
    stored_mapping = get_schema_mapping() if get_schema_mapping_signature() == dataset_signature else None
    base_mapping = merge_schema_mapping(data, stored_mapping)

    st.subheader("Dataset Schema Wizard")
    st.caption(
        "Map your dataset fields to dashboard roles so prediction, clustering, KPIs, reports, and preprocessing can adapt to custom academic schemas."
    )

    status_table = build_schema_status_table(data, base_mapping)
    st.dataframe(status_table, width="stretch", hide_index=True)

    with st.expander("Adjust schema mapping", expanded=True):
        top_left, top_right = st.columns(2)
        with top_left:
            student_id_column = st.selectbox(
                "Student ID column",
                options=["None", *all_columns],
                index=(["None", *all_columns].index(base_mapping.get("student_id_column")) if base_mapping.get("student_id_column") in all_columns else 0),
                key="schema_student_id_column",
            )
            target_column = st.selectbox(
                "Target column",
                options=["Derive from score columns", *all_columns],
                index=(["Derive from score columns", *all_columns].index(base_mapping.get("target_column")) if base_mapping.get("target_column") in all_columns else 0),
                key="schema_target_column",
            )
            attendance_column = st.selectbox(
                "Attendance column",
                options=["None", *all_columns],
                index=(["None", *all_columns].index(base_mapping.get("attendance_column")) if base_mapping.get("attendance_column") in all_columns else 0),
                key="schema_attendance_column",
            )
        with top_right:
            score_columns = st.multiselect(
                "Score columns",
                options=numeric_columns,
                default=[column for column in base_mapping.get("score_columns", []) if column in numeric_columns],
                key="schema_score_columns",
                help="These columns drive score KPIs and can also be used to derive Pass/Fail when no direct target column exists.",
            )
            risk_indicator_columns = st.multiselect(
                "Risk indicator columns",
                options=all_columns,
                default=[column for column in base_mapping.get("risk_indicator_columns", []) if column in all_columns],
                key="schema_risk_indicator_columns",
                help="Examples include attendance, absences, study habits, behavior, or past failures.",
            )

        prediction_mapping = dict(base_mapping.get("prediction_feature_mapping", {}))
        st.markdown("**Prediction input mapping**")
        prediction_columns = st.columns(2)
        prediction_fields = [
            ("study_time", "Study time"),
            ("absences", "Absences"),
            ("failures", "Failures"),
            ("previous_grade_1", "Previous score 1"),
            ("previous_grade_2", "Previous score 2"),
        ]
        for index, (feature_key, feature_label) in enumerate(prediction_fields):
            with prediction_columns[index % 2]:
                prediction_mapping[feature_key] = st.selectbox(
                    feature_label,
                    options=["None", *numeric_columns],
                    index=(["None", *numeric_columns].index(prediction_mapping.get(feature_key)) if prediction_mapping.get(feature_key) in numeric_columns else 0),
                    key=f"schema_prediction_{feature_key}",
                )

        clustering_left, clustering_right = st.columns(2)
        with clustering_left:
            classification_features = st.multiselect(
                "Classification feature columns",
                options=all_columns,
                default=[column for column in base_mapping.get("classification_feature_columns", []) if column in all_columns],
                key="schema_classification_feature_columns",
            )
            clustering_feature_columns = st.multiselect(
                "Clustering feature columns",
                options=numeric_columns,
                default=[column for column in base_mapping.get("clustering_feature_columns", []) if column in numeric_columns],
                key="schema_clustering_feature_columns",
            )
        with clustering_right:
            performance_columns = st.multiselect(
                "Performance columns for cluster labels",
                options=numeric_columns,
                default=[column for column in base_mapping.get("performance_columns", []) if column in numeric_columns],
                key="schema_performance_columns",
            )

    schema_mapping = merge_schema_mapping(
        data,
        {
            "student_id_column": None if student_id_column == "None" else student_id_column,
            "target_column": None if target_column == "Derive from score columns" else target_column,
            "attendance_column": None if attendance_column == "None" else attendance_column,
            "score_columns": score_columns,
            "risk_indicator_columns": risk_indicator_columns,
            "prediction_feature_mapping": {
                key: value
                for key, value in prediction_mapping.items()
                if value is not None and value != "None"
            },
            "classification_feature_columns": classification_features,
            "clustering_feature_columns": clustering_feature_columns,
            "performance_columns": performance_columns,
        },
    )

    support_messages = []
    support_messages.append("Prediction ready" if mapping_supports_prediction(schema_mapping) else "Prediction needs more mapped inputs")
    support_messages.append("Classification ready" if mapping_supports_classification(schema_mapping) else "Classification needs a target or score columns")
    support_messages.append("Clustering ready" if mapping_supports_clustering(schema_mapping) else "Clustering needs numeric clustering/performance columns")
    st.caption(" | ".join(support_messages))

    return schema_mapping


def render_column_overview(profile: DatasetProfile) -> None:
    """Display column names, data types, and missing-value details."""
    left, right = st.columns([0.95, 1.05])

    with left:
        st.subheader("Column Names")
        st.write(", ".join(profile.column_names))

    with right:
        st.subheader("Data Types")
        st.dataframe(profile.data_types, width="stretch", hide_index=True)

    st.subheader("Missing Values")
    st.dataframe(profile.missing_values, width="stretch", hide_index=True)


def render_dataset_preview(data: pd.DataFrame) -> None:
    """Render an interactive preview with row count and optional column filtering."""
    st.subheader("Dataset Preview")

    # Let users preview a manageable number of rows without overwhelming the UI.
    max_preview_rows = min(len(data), 100)
    preview_rows = st.slider(
        "Rows to preview",
        min_value=5,
        max_value=max_preview_rows if max_preview_rows >= 5 else 5,
        value=min(10, max_preview_rows) if max_preview_rows else 5,
        step=5,
        disabled=data.empty,
    )

    selected_columns = st.multiselect(
        "Columns to display",
        options=list(data.columns),
        default=list(data.columns),
    )

    # Fall back to all columns if a user clears the multiselect.
    preview_columns = selected_columns if selected_columns else list(data.columns)
    st.dataframe(data.loc[:, preview_columns].head(preview_rows), width="stretch")


def render_summary_statistics(profile: DatasetProfile) -> None:
    """Display summary statistics for numeric and categorical fields."""
    st.subheader("Summary Statistics")
    st.dataframe(profile.summary_statistics, width="stretch", hide_index=True)


def render_dataset_profile_tabs(data: pd.DataFrame, profile: DatasetProfile) -> None:
    """Render preview and inspection tabs for the active dataset."""
    tabs = st.tabs(["Preview", "Schema Mapping", "Columns", "Missing Values", "Summary Statistics"])
    with tabs[0]:
        render_dataset_preview(data)
    with tabs[1]:
        dataset_signature = get_dataset_signature()
        if dataset_signature is not None:
            schema_mapping = render_schema_detection_wizard(data, dataset_signature)
            if get_schema_mapping_signature() != dataset_signature or get_schema_mapping() != schema_mapping:
                set_schema_mapping(schema_mapping, dataset_signature)
                clear_analytics_state()
    with tabs[2]:
        render_column_overview(profile)
    with tabs[3]:
        st.subheader("Missing Value Report")
        st.dataframe(profile.missing_values, width="stretch", hide_index=True)
    with tabs[4]:
        render_summary_statistics(profile)


def process_uploaded_dataset_file(
    uploaded_file: Any,
    *,
    allowed_file_types: list[str] | None = None,
    widget_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Read, validate, store, and auto-map one uploaded dataset file.

    This helper keeps upload handling consistent across the Home page and the
    dedicated inspection workspace. It stores the raw dataset in session state,
    persists uploads to SQLite, and auto-detects an initial schema mapping so
    the rest of the dashboard can react immediately.
    """
    allowed_file_types = allowed_file_types or SUPPORTED_FILE_TYPES
    widget_keys = widget_keys or DATASET_DEPENDENT_WIDGET_KEYS

    upload_errors = validate_uploaded_file(uploaded_file, allowed_file_types)
    if upload_errors:
        return {"errors": upload_errors}

    try:
        uploaded_file.seek(0)
        data = read_uploaded_dataset(uploaded_file)
    except EmptyDataError:
        return {"error": "The uploaded CSV file is empty or does not contain readable columns."}
    except ParserError:
        return {"error": "The CSV file could not be parsed. Check delimiters, quotes, and malformed rows."}
    except ImportError:
        return {"error": "A file reader dependency is not available. Install dependencies from requirements.txt and try again."}
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        return {
            "error": "Unable to read the uploaded file. Confirm that it is a valid CSV, Excel, or table-based PDF file.",
            "error_detail": str(error),
        }

    if data.empty:
        return {
            "warning": "The uploaded file was read successfully, but it does not contain any rows.",
            "data": data,
            "dataset_name": uploaded_file.name,
        }

    dataframe_errors = validate_dataframe_for_dashboard(data)
    if dataframe_errors:
        return {"errors": dataframe_errors}

    upload_signature = build_uploaded_file_signature(uploaded_file, data)
    if get_dataset_signature() != upload_signature:
        clear_derived_state(widget_keys=widget_keys)

    set_raw_dataset(
        data,
        dataset_name=uploaded_file.name,
        source="upload",
        signature=upload_signature,
        reset_derived=False,
    )

    existing_mapping = get_schema_mapping() if get_schema_mapping_signature() == upload_signature else None
    detected_mapping = merge_schema_mapping(data, existing_mapping)
    set_schema_mapping(detected_mapping, upload_signature)

    result: dict[str, Any] = {
        "data": data,
        "dataset_name": uploaded_file.name,
        "upload_signature": upload_signature,
        "schema_mapping": detected_mapping,
    }

    if st.session_state.get("last_saved_upload_signature") != upload_signature:
        try:
            dataset_id = save_dataset_upload(uploaded_file.name, data)
            st.session_state["last_saved_upload_signature"] = upload_signature
            st.session_state["last_saved_dataset_id"] = dataset_id
            result["saved_dataset_id"] = dataset_id
            try:
                record_etl_job(uploaded_file.name, "extract", rows_in=0, rows_out=len(data))
                record_etl_job(uploaded_file.name, "validate", rows_in=len(data), rows_out=len(data))
                record_audit_event("dataset_upload", entity_name=uploaded_file.name, detail=f"{len(data):,} rows, {len(data.columns)} columns", rows_affected=len(data))
            except Exception:
                pass
        except Exception as error:
            result["save_warning"] = "The dataset was loaded, but it could not be saved to SQLite."
            result["save_error_detail"] = str(error)
    elif "last_saved_dataset_id" in st.session_state:
        result["saved_dataset_id"] = st.session_state["last_saved_dataset_id"]

    return result


def render_dataset_upload_module() -> pd.DataFrame | None:
    """Render the full upload module and return the uploaded dataframe when available."""
    st.markdown(
        """
        <div class="module-intro">
            <strong>Upload a student dataset</strong>
            <span>Supported formats: CSV, XLSX, XLS, and table-based PDF.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_data_governance_indicators()

    uploaded_file = st.file_uploader(
        "Choose a dataset file",
        type=SUPPORTED_FILE_TYPES,
        help="Upload a CSV, Excel, or table-based PDF file containing student performance records.",
    )

    if uploaded_file is None:
        active_data = get_raw_dataset()
        active_name = get_dataset_name()
        if isinstance(active_data, pd.DataFrame) and not active_data.empty:
            st.info(f"Using active dataset `{active_name}`. Upload another file to replace it.")
            profile = build_dataset_profile(active_data)
            render_overview_metrics(profile)
            render_dataset_profile_tabs(active_data, profile)
            return active_data

        st.info("Upload a file to preview records and inspect dataset quality.")
        return None

    result = process_uploaded_dataset_file(uploaded_file, allowed_file_types=SUPPORTED_FILE_TYPES)
    if result.get("errors"):
        display_validation_errors(result["errors"])
        return None
    if result.get("error"):
        st.error(result["error"])
        if result.get("error_detail"):
            st.caption(f"Technical detail: {result['error_detail']}")
        return None

    data = result.get("data")
    if not isinstance(data, pd.DataFrame):
        return None
    if data.empty:
        st.warning(str(result.get("warning", "The uploaded file was read successfully, but it does not contain any rows.")))
        return data

    st.success(f"Loaded `{result['dataset_name']}` successfully.")
    if result.get("saved_dataset_id") is not None:
        st.caption(f"Dataset available in SQLite with ID `{result['saved_dataset_id']}`.")
    if result.get("save_warning"):
        st.warning(str(result["save_warning"]))
        if result.get("save_error_detail"):
            st.caption(f"Technical detail: {result['save_error_detail']}")

    profile = build_dataset_profile(data)

    render_overview_metrics(profile)
    render_dataset_profile_tabs(data, profile)

    return data
