"""Reusable Pandas helpers for exploratory data analysis.

These functions avoid Streamlit-specific code so the same analysis logic can be
used by charts, reports, tests, or future preprocessing workflows.
"""

from __future__ import annotations

import pandas as pd


def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return columns that can be used for numeric charts and correlations."""
    return data.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return columns that work well for frequency tables and category charts."""
    return data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def get_datetime_columns(data: pd.DataFrame) -> list[str]:
    """Return columns already recognized by Pandas as datetime values."""
    return data.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def build_summary_statistics(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Generate summary statistics for the selected columns."""
    selected_data = data.loc[:, columns] if columns else data
    summary = selected_data.describe(include="all").transpose().reset_index()
    return summary.rename(columns={"index": "Column"})


def build_frequency_table(data: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    """Create a frequency table for a selected column."""
    # Convert values to strings so missing values and mixed data types display consistently.
    values = data[column].astype("string").fillna("Missing")
    frequency = values.value_counts(dropna=False).head(top_n).rename_axis(column).reset_index(name="Frequency")
    frequency["Percentage"] = ((frequency["Frequency"] / len(data)) * 100).round(2)
    return frequency


def build_correlation_matrix(
    data: pd.DataFrame,
    columns: list[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate correlations for selected numeric columns."""
    if len(columns) < 2:
        return pd.DataFrame()
    return data.loc[:, columns].corr(method=method)
