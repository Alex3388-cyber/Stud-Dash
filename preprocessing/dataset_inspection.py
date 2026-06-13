"""Dataset inspection helpers for uploaded student performance files.

These functions keep Pandas profiling logic separate from Streamlit layout code,
which makes the upload module easier to test and reuse in other dashboard pages.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """Simple container for the most important dataset overview values."""

    row_count: int
    column_count: int
    column_names: list[str]
    data_types: pd.DataFrame
    missing_values: pd.DataFrame
    summary_statistics: pd.DataFrame


def build_dataset_profile(data: pd.DataFrame) -> DatasetProfile:
    """Create a reusable profile object from a Pandas dataframe."""
    # Convert dtype information into a dataframe so it can be displayed cleanly.
    data_types = (
        data.dtypes.astype(str)
        .rename("Data Type")
        .reset_index()
        .rename(columns={"index": "Column"})
    )

    # Missing values are shown as both counts and percentages for quick review.
    missing_values = pd.DataFrame(
        {
            "Column": data.columns,
            "Missing Values": data.isna().sum().values,
            "Missing Percentage": (data.isna().mean().values * 100).round(2),
        }
    ).sort_values("Missing Values", ascending=False)

    # Include numeric and categorical columns in the summary table.
    summary_statistics = data.describe(include="all").transpose().reset_index()
    summary_statistics = summary_statistics.rename(columns={"index": "Column"})

    return DatasetProfile(
        row_count=len(data),
        column_count=len(data.columns),
        column_names=list(data.columns),
        data_types=data_types,
        missing_values=missing_values,
        summary_statistics=summary_statistics,
    )
