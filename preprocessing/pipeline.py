"""Reusable preprocessing and feature engineering functions."""

import pandas as pd


SCORE_COLUMNS = ["math_score", "reading_score", "writing_score"]


def clean_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the student dataset."""
    cleaned = data.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower().str.replace(" ", "_")
    return cleaned.drop_duplicates()


def add_target_average_score(data: pd.DataFrame) -> pd.DataFrame:
    """Add the average score target used by the starter regression model."""
    transformed = clean_student_data(data)
    transformed["average_score"] = transformed[SCORE_COLUMNS].mean(axis=1)
    return transformed
