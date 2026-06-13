"""Service wrapper for K-Means clustering workflows."""

from __future__ import annotations

import pandas as pd

from models.clustering import run_kmeans_clustering


def run_clustering(
    data: pd.DataFrame,
    feature_columns: list[str],
    performance_columns: list[str],
    random_state: int = 42,
):
    """Run the dashboard K-Means clustering engine."""
    return run_kmeans_clustering(
        data=data,
        feature_columns=feature_columns,
        performance_columns=performance_columns,
        random_state=random_state,
    )
