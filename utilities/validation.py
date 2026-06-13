"""Validation helpers for safer Streamlit user interactions."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd


MAX_UPLOAD_SIZE_MB = 50


def validate_uploaded_file(uploaded_file: Any, allowed_extensions: list[str]) -> list[str]:
    """Validate an uploaded file before Pandas attempts to parse it."""
    errors: list[str] = []
    file_name = getattr(uploaded_file, "name", "")
    suffix = Path(file_name).suffix.lower().lstrip(".")

    if not file_name:
        errors.append("The uploaded file does not have a valid file name.")

    if suffix not in allowed_extensions:
        errors.append("Please upload a supported file type: CSV, XLSX, XLS, or PDF.")

    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        try:
            file_size = len(uploaded_file.getbuffer())
        except Exception:
            file_size = 0

    if file_size == 0:
        errors.append("The uploaded file is empty. Choose a file with student records.")

    max_size_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        errors.append(f"The uploaded file is too large. Maximum supported size is {MAX_UPLOAD_SIZE_MB} MB.")

    return errors


def validate_dataframe_for_dashboard(data: pd.DataFrame | None) -> list[str]:
    """Validate a dataframe before dashboard modules use it."""
    if data is None:
        return ["No dataset is available. Upload a dataset first."]

    errors: list[str] = []
    if len(data.columns) == 0:
        errors.append("The dataset does not contain any columns.")

    blank_columns = [column for column in data.columns if str(column).strip() == ""]
    if blank_columns:
        errors.append("The dataset contains blank column names. Rename those columns and upload again.")

    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        errors.append(f"The dataset contains duplicate column names: {', '.join(map(str, duplicated))}.")

    return errors


def validate_prediction_inputs(
    inputs: dict[str, float],
    field_constraints: dict[str, dict[str, float | str]] | None = None,
) -> list[str]:
    """Validate the student prediction form values."""
    errors: list[str] = []
    ranges = field_constraints or {
        "study_time": (1.0, 4.0, "Study Time"),
        "absences": (0.0, 100.0, "Absences"),
        "failures": (0.0, 10.0, "Failures"),
        "previous_grade_1": (0.0, 20.0, "Previous Grade G1"),
        "previous_grade_2": (0.0, 20.0, "Previous Grade G2"),
    }

    for field, config in ranges.items():
        if isinstance(config, dict):
            minimum = float(config.get("min", 0.0))
            maximum = float(config.get("max", 100.0))
            label = str(config.get("title", field.replace("_", " ").title()))
        else:
            minimum, maximum, label = config
        value = inputs.get(field)
        if value is None or pd.isna(value):
            errors.append(f"{label} is required.")
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue

        if not isfinite(numeric_value):
            errors.append(f"{label} must be a finite number.")
        elif numeric_value < minimum or numeric_value > maximum:
            errors.append(f"{label} must be between {minimum:g} and {maximum:g}.")

    return errors


def display_validation_errors(errors: list[str]) -> None:
    """Render validation errors consistently in Streamlit modules."""
    import streamlit as st

    for error in errors:
        st.error(error)
