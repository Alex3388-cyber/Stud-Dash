"""Service wrapper for classification model training and reporting."""

from __future__ import annotations

import pandas as pd

from models.classification import build_metrics_table, train_classification_models


def train_models(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str | None = None,
    score_columns: list[str] | None = None,
    pass_threshold: float = 60.0,
    test_size: float = 0.3,
    random_state: int = 42,
    cv_folds: int = 5,
):
    """Train dashboard classification models through a service interface."""
    return train_classification_models(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        score_columns=score_columns,
        pass_threshold=pass_threshold,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
    )


def build_model_metrics_table(results):
    """Return the model comparison table."""
    return build_metrics_table(results)
