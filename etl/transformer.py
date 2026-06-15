"""ETL transformer — cleans and normalises raw extracted data."""

from __future__ import annotations

import pandas as pd


def transform(data: pd.DataFrame) -> pd.DataFrame:
    """Clean, deduplicate, and normalise a raw DataFrame."""
    if data.empty:
        return data

    # Strip whitespace from column names
    data = data.copy()
    data.columns = data.columns.str.strip()

    # Remove fully empty rows
    data = data.dropna(how="all").reset_index(drop=True)

    # Remove exact duplicate rows
    data = data.drop_duplicates().reset_index(drop=True)

    # Coerce columns that are mostly numeric but stored as strings
    for col in data.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(data[col], errors="coerce")
        if converted.notna().mean() > 0.8:
            data[col] = converted

    # Normalise string columns: strip leading/trailing whitespace
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].str.strip()

    return data
