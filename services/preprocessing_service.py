"""Service wrapper for preprocessing workflows."""

from __future__ import annotations

import pandas as pd

from preprocessing.preprocessor import build_preprocessing_summary_table, preprocess_dataset


def run_preprocessing(data: pd.DataFrame, cardinality_threshold: int = 30):
    """Run preprocessing through a stable service interface."""
    return preprocess_dataset(data, cardinality_threshold=cardinality_threshold)


def build_summary_table(summary):
    """Build the preprocessing summary table."""
    return build_preprocessing_summary_table(summary)
