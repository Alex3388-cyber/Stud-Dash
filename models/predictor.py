"""Prediction helpers for trained student performance models."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "student_score_pipeline.joblib"


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a serialized Scikit-learn pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Train a model before running predictions."
        )
    return joblib.load(model_path)


def predict_average_score(features: pd.DataFrame, model_path: Path = DEFAULT_MODEL_PATH) -> pd.Series:
    """Predict average student score from a feature dataframe."""
    model = load_model(model_path)
    predictions = model.predict(features)
    return pd.Series(predictions, name="predicted_average_score")
