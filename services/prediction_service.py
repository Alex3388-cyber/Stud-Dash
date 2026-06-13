"""Service wrapper for live prediction model preparation and inference."""

from __future__ import annotations

import pandas as pd

from models.student_prediction import (
    infer_prediction_feature_mapping,
    predict_student_performance,
    train_logistic_prediction_model,
)


def infer_feature_mapping(data: pd.DataFrame, schema_mapping: dict[str, object] | None = None):
    """Infer the active prediction feature mapping."""
    return infer_prediction_feature_mapping(data, schema_mapping=schema_mapping)


def prepare_prediction_model(data: pd.DataFrame, schema_mapping: dict[str, object] | None = None):
    """Prepare the logistic regression prediction model bundle."""
    return train_logistic_prediction_model(data, schema_mapping=schema_mapping)


def run_prediction(model_bundle, **inputs):
    """Run a live student prediction."""
    return predict_student_performance(model_bundle=model_bundle, **inputs)
