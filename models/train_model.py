"""Starter Scikit-learn training workflow for student performance prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing.pipeline import add_target_average_score


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "student_score_pipeline.joblib"

CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "lunch_type",
    "test_preparation_course",
]
NUMERIC_FEATURES = ["study_time_hours", "attendance_rate"]
TARGET = "average_score"


def build_training_pipeline() -> Pipeline:
    """Create a preprocessing and regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )


def train_model(data: pd.DataFrame | None = None) -> dict[str, float]:
    """Train and persist a model from an explicitly provided dataset."""
    if data is None:
        raise ValueError("A real uploaded dataset must be provided. Sample-data fallback is disabled.")

    training_data = add_target_average_score(data)
    features = training_data[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    target = training_data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    pipeline = build_training_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "mean_absolute_error": float(mean_absolute_error(y_test, predictions)),
        "r2_score": float(r2_score(y_test, predictions)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    print(train_model())
