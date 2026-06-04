"""Student Pass/Fail prediction helpers using Logistic Regression.

The live prediction console uses five semantic academic inputs:

- study time
- absences or attendance proxy
- failures or academic history proxy
- previous score 1
- previous score 2

The physical dataset columns for those inputs come from the active schema
mapping so the dashboard can work with different academic dataset formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.classification import FAIL_LABEL, PASS_LABEL, create_pass_fail_target_from_scores, normalize_pass_fail_target
from utilities.schema_mapping import normalize_lookup


FORM_FEATURES = ["study_time", "absences", "failures", "previous_grade_1", "previous_grade_2"]
FEATURE_LABELS = {
    "study_time": "Study Time",
    "absences": "Absences or Attendance",
    "failures": "Failures",
    "previous_grade_1": "Previous Grade G1",
    "previous_grade_2": "Previous Grade G2",
}


@dataclass(frozen=True)
class PredictionModelBundle:
    """Trained Logistic Regression model plus metadata needed by the UI."""

    model: Pipeline
    training_rows: int
    feature_mapping: dict[str, str]
    feature_modes: dict[str, str]
    target_source: str


@dataclass(frozen=True)
class StudentPrediction:
    """Prediction result shown to the user."""

    predicted_label: str
    pass_probability: float
    fail_probability: float
    confidence_score: float
    risk_level: str
    recommendation: str


def infer_prediction_feature_mapping(
    data: pd.DataFrame,
    schema_mapping: dict[str, object] | None = None,
) -> dict[str, str]:
    """Map the dashboard prediction inputs to columns using the schema mapping first."""
    schema_mapping = schema_mapping or {}
    mapped = schema_mapping.get("prediction_feature_mapping", {})
    resolved_mapping = {}
    if isinstance(mapped, dict) and mapped:
        resolved_mapping = {
            feature: column
            for feature, column in mapped.items()
            if column in data.columns
        }
    if len(resolved_mapping) == len(FORM_FEATURES):
        return resolved_mapping

    from utilities.schema_mapping import detect_prediction_feature_mapping

    schema_score_columns = [
        column for column in schema_mapping.get("score_columns", []) or []
        if column in data.columns
    ]
    schema_attendance_column = schema_mapping.get("attendance_column")
    detected_mapping = detect_prediction_feature_mapping(
        data,
        score_columns=schema_score_columns,
        attendance_column=str(schema_attendance_column) if schema_attendance_column in data.columns else None,
    )
    if resolved_mapping:
        detected_mapping.update(resolved_mapping)
    return detected_mapping


def find_pass_fail_column(data: pd.DataFrame, schema_mapping: dict[str, object] | None = None) -> str | None:
    """Find an existing Pass/Fail target column from the active schema mapping."""
    schema_mapping = schema_mapping or {}
    target_column = schema_mapping.get("target_column")
    return str(target_column) if target_column in data.columns else None


def choose_score_columns_for_target(
    data: pd.DataFrame,
    schema_mapping: dict[str, object] | None,
    feature_mapping: dict[str, str],
) -> list[str]:
    """Choose mapped performance or score columns for deriving Pass/Fail."""
    schema_mapping = schema_mapping or {}
    performance_columns = [
        column for column in schema_mapping.get("performance_columns", []) or []
        if column in data.columns
    ]
    if performance_columns:
        return performance_columns

    score_columns = [
        column for column in schema_mapping.get("score_columns", []) or []
        if column in data.columns
    ]
    if score_columns:
        return score_columns[:2]

    mapped_score_columns = [
        feature_mapping[feature]
        for feature in ["previous_grade_1", "previous_grade_2"]
        if feature in feature_mapping
    ]
    return mapped_score_columns


def get_absence_feature_mode(values: pd.Series, mapped_column: str) -> str:
    """Describe how one absence-like feature should be interpreted."""
    normalized_column = normalize_lookup(mapped_column)
    if "attendance" not in normalized_column:
        return "absence_count"

    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return "attendance_percent_proxy"

    max_value = float(clean_values.max())
    if max_value <= 1.5:
        return "attendance_ratio_proxy"
    return "attendance_percent_proxy"


def transform_absence_feature(values: pd.Series, feature_mode: str) -> pd.Series:
    """Convert attendance proxies into a risk-style absence feature for modeling."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    if feature_mode == "attendance_ratio_proxy":
        return (1 - numeric_values.clip(lower=0, upper=1)) * 100
    if feature_mode == "attendance_percent_proxy":
        return 100 - numeric_values.clip(lower=0, upper=100)
    return numeric_values


def build_prediction_features(
    data: pd.DataFrame,
    feature_mapping: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], list[str]]:
    """Build model-ready prediction features plus field metadata.

    The prediction console can work with a direct absences column or an
    attendance-style proxy. When no failure-history column exists, a neutral
    zero-valued feature is added so the rest of the console can still run.
    """
    features = pd.DataFrame(index=data.index)
    resolved_feature_mapping: dict[str, str] = {}
    feature_modes: dict[str, str] = {}
    missing_required_features: list[str] = []

    for feature in FORM_FEATURES:
        mapped_column = feature_mapping.get(feature)

        if mapped_column is None:
            if feature == "failures":
                features[feature] = pd.Series(0.0, index=data.index, dtype="float64")
                resolved_feature_mapping[feature] = "Default (0)"
                feature_modes[feature] = "default_zero"
                continue

            missing_required_features.append(feature)
            continue

        resolved_feature_mapping[feature] = mapped_column
        if feature == "absences":
            raw_values = data[mapped_column]
            feature_mode = get_absence_feature_mode(raw_values, mapped_column)
            features[feature] = transform_absence_feature(raw_values, feature_mode)
            feature_modes[feature] = feature_mode
        else:
            features[feature] = pd.to_numeric(data[mapped_column], errors="coerce")
            feature_modes[feature] = "direct"

    return features, resolved_feature_mapping, feature_modes, missing_required_features


def build_logistic_regression_pipeline() -> Pipeline:
    """Build the preprocessing and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            # Missing form features are filled with medians learned from the
            # training data so prediction can continue when the source dataset
            # contains incomplete records.
            ("imputer", SimpleImputer(strategy="median")),
            # Logistic Regression is sensitive to feature scale. StandardScaler
            # puts study time, absences, failures, and grade values on
            # comparable scales before the model learns decision boundaries.
            ("scaler", StandardScaler()),
            # Logistic Regression learns how the four academic features relate
            # to the Pass/Fail target and can output class probabilities.
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_logistic_prediction_model(
    data: pd.DataFrame,
    schema_mapping: dict[str, object] | None = None,
    pass_threshold: float = 60.0,
) -> PredictionModelBundle:
    """Train a Logistic Regression model for the prediction form."""
    feature_mapping = infer_prediction_feature_mapping(data, schema_mapping=schema_mapping)
    features, resolved_feature_mapping, feature_modes, missing_features = build_prediction_features(data, feature_mapping)
    if missing_features:
        readable_missing = ", ".join(FEATURE_LABELS[feature] for feature in missing_features)
        raise ValueError(f"The dataset is missing required prediction fields: {readable_missing}.")

    target_column = find_pass_fail_column(data, schema_mapping=schema_mapping)
    if target_column:
        target = normalize_pass_fail_target(data[target_column])
        target_source = target_column
    else:
        score_columns = choose_score_columns_for_target(data, schema_mapping, feature_mapping)
        score_values = data.loc[:, score_columns].apply(pd.to_numeric, errors="coerce") if score_columns else pd.DataFrame()
        max_score = score_values.stack().max() if not score_values.empty else pd.NA
        if max_score == max_score and float(max_score) <= 1.5:
            pass_threshold = 0.6
        elif max_score == max_score and float(max_score) <= 20:
            pass_threshold = 10.0
        target = create_pass_fail_target_from_scores(data, score_columns, pass_threshold)
        target_source = f"Derived from {', '.join(score_columns)} at threshold {pass_threshold:g}"

    training_data = features.copy()
    training_data["target"] = target
    training_data = training_data.dropna(subset=["target"]).reset_index(drop=True)
    target = training_data["target"]
    features = training_data[FORM_FEATURES]

    if target.nunique() < 2:
        raise ValueError("The training target must contain both Pass and Fail records.")

    model = build_logistic_regression_pipeline()
    model.fit(features, target)

    return PredictionModelBundle(
        model=model,
        training_rows=len(features),
        feature_mapping=resolved_feature_mapping,
        feature_modes=feature_modes,
        target_source=target_source,
    )


def get_risk_level(pass_probability: float) -> str:
    """Translate pass probability into a student risk level."""
    if pass_probability >= 0.75:
        return "Low Risk"
    if pass_probability >= 0.5:
        return "Moderate Risk"
    return "High Risk"


def get_recommendation(
    predicted_label: str,
    risk_level: str,
    study_time: float,
    absences: float,
    failures: float,
    previous_grade_average: float,
) -> str:
    """Return a recommendation using both model output and input risk signals."""
    interventions: list[str] = []

    if absences >= 10:
        interventions.append("reduce absenteeism and restore regular class participation")
    if failures >= 2:
        interventions.append("prioritize remediation in previously failed subjects")
    if previous_grade_average < 10:
        interventions.append("strengthen revision on weak topics before the next assessment")
    if study_time <= 1:
        interventions.append("increase weekly study time with a structured revision plan")

    if predicted_label == PASS_LABEL and risk_level == "Low Risk":
        base_message = "The student is trending positively and is likely to maintain a passing outcome."
    elif predicted_label == PASS_LABEL:
        base_message = "The student is currently on a passing path, but the academic profile still needs monitoring."
    elif risk_level == "Moderate Risk":
        base_message = "The student needs targeted academic support to avoid slipping into a failing outcome."
    else:
        base_message = "The student is at high risk of failing and needs immediate academic intervention."

    if not interventions:
        return f"{base_message} Continue guided practice, weekly review, and close progress tracking."

    return f"{base_message} Recommended action: {', '.join(interventions[:3])}."


def predict_student_performance(
    model_bundle: PredictionModelBundle,
    study_time: float,
    absences: float,
    failures: float,
    previous_grade_1: float,
    previous_grade_2: float,
) -> StudentPrediction:
    """Predict Pass/Fail and probability for one student form submission."""
    for label, value in {
        "study_time": study_time,
        "absences": absences,
        "failures": failures,
        "previous_grade_1": previous_grade_1,
        "previous_grade_2": previous_grade_2,
    }.items():
        if value is None or pd.isna(value):
            raise ValueError(f"{label} is required.")
        if not isfinite(float(value)):
            raise ValueError(f"{label} must be a finite number.")

    input_data = pd.DataFrame(
        [
            {
                "study_time": study_time,
                "absences": absences,
                "failures": failures,
                "previous_grade_1": previous_grade_1,
                "previous_grade_2": previous_grade_2,
            }
        ]
    )

    predicted_label = str(model_bundle.model.predict(input_data)[0])
    class_probabilities = model_bundle.model.predict_proba(input_data)[0]
    class_lookup = dict(zip(model_bundle.model.classes_, class_probabilities))

    pass_probability = float(class_lookup.get(PASS_LABEL, 0.0))
    fail_probability = float(class_lookup.get(FAIL_LABEL, 0.0))
    confidence_score = max(pass_probability, fail_probability)
    risk_level = get_risk_level(pass_probability)
    previous_grade_average = (float(previous_grade_1) + float(previous_grade_2)) / 2

    return StudentPrediction(
        predicted_label=predicted_label,
        pass_probability=pass_probability,
        fail_probability=fail_probability,
        confidence_score=confidence_score,
        risk_level=risk_level,
        recommendation=get_recommendation(
            predicted_label,
            risk_level,
            float(study_time),
            float(absences),
            float(failures),
            previous_grade_average,
        ),
    )
