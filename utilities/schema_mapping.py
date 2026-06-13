"""Dynamic dataset schema detection and mapping helpers.

This service lets the dashboard adapt to different academic dataset schemas by
separating semantic roles from physical column names. Modules can ask for
meaningful fields such as score columns, target column, attendance column, or
risk indicators without assuming UCI naming.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


SchemaMapping = dict[str, Any]


def normalize_lookup(name: str) -> str:
    """Normalize column names for flexible matching."""
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace("/", "")


def numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Convert one dataframe column to numeric values when possible."""
    return pd.to_numeric(data[column], errors="coerce")


def has_numeric_values(data: pd.DataFrame, column: str) -> bool:
    """Return True when a column contains at least one numeric value."""
    return numeric_series(data, column).notna().any()


def find_matching_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first dataset column matching one of the candidate names."""
    normalized_columns = {normalize_lookup(column): column for column in data.columns}
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def find_column_by_keywords(
    data: pd.DataFrame,
    include_keywords: list[str],
    *,
    exclude_keywords: list[str] | None = None,
    require_numeric: bool = False,
) -> str | None:
    """Find a column whose normalized name contains the requested keywords."""
    exclude_keywords = exclude_keywords or []

    for column in data.columns:
        normalized = normalize_lookup(column)
        if any(keyword not in normalized for keyword in include_keywords):
            continue
        if any(keyword in normalized for keyword in exclude_keywords):
            continue
        if require_numeric and not has_numeric_values(data, column):
            continue
        return column
    return None


def get_student_id_column(data: pd.DataFrame) -> str | None:
    """Find a likely student identifier column."""
    exact_candidates = ["studentid", "studentnumber", "studentno", "matricnumber", "registrationnumber"]
    matched = find_matching_column(data, exact_candidates)
    if matched:
        return matched

    for column in data.columns:
        normalized = normalize_lookup(column)
        if "student" in normalized and ("id" in normalized or "number" in normalized or "no" in normalized):
            return column
    return None


def detect_target_column(data: pd.DataFrame) -> str | None:
    """Detect a direct Pass/Fail-style target column when available."""
    priority_names = ["passfail", "passstatus", "result", "status", "outcome", "performance"]
    blocked_names = {"failures", "failure", "failedsubjects", "pastfailures", "previousfailures"}
    candidate_columns = sorted(
        data.columns,
        key=lambda column: 0 if normalize_lookup(column) in priority_names else 1,
    )

    pass_values = {"pass", "passed", "yes", "y", "true", "1", "successful"}
    fail_values = {"fail", "failed", "no", "n", "false", "0", "unsuccessful"}

    for column in candidate_columns:
        normalized_name = normalize_lookup(column)
        if normalized_name in blocked_names:
            continue
        if normalized_name not in priority_names and "pass" not in normalized_name and "fail" not in normalized_name:
            continue

        normalized_values = data[column].astype("string").str.strip().str.lower().dropna()
        if normalized_values.empty:
            continue
        valid_values = normalized_values[normalized_values.isin(pass_values | fail_values)]
        if len(valid_values) / max(len(normalized_values), 1) >= 0.5:
            return column
    return None


def detect_attendance_column(data: pd.DataFrame) -> str | None:
    """Detect an attendance or absence field."""
    attendance_candidates = [
        "attendance",
        "attendancerate",
        "attendancepercentage",
        "attendancepercent",
        "attendance_rate",
        "attendance_percent",
    ]
    attendance_column = find_matching_column(data, attendance_candidates)
    if attendance_column and has_numeric_values(data, attendance_column):
        return attendance_column

    absence_candidates = ["absences", "absence", "classesmissed", "missedclasses", "missed_classes"]
    absence_column = find_matching_column(data, absence_candidates)
    if absence_column and has_numeric_values(data, absence_column):
        return absence_column

    for column in data.columns:
        normalized = normalize_lookup(column)
        if ("attendance" in normalized or "absence" in normalized or "missed" in normalized) and has_numeric_values(data, column):
            return column
    return None


def detect_score_columns(data: pd.DataFrame) -> list[str]:
    """Detect columns that represent academic scores or grades."""
    score_keywords = [
        "g1",
        "g2",
        "g3",
        "score",
        "mark",
        "grade",
        "exam",
        "test",
        "assignment",
        "midsem",
        "midterm",
        "final",
        "result",
        "quiz",
        "coursework",
    ]
    excluded_keywords = ["id", "attendance", "study", "hour", "cluster", "risk", "target", "pass", "fail"]

    score_columns: list[str] = []
    for column in data.columns:
        normalized = normalize_lookup(column)
        if any(keyword in normalized for keyword in excluded_keywords):
            continue
        if any(keyword in normalized for keyword in score_keywords) and has_numeric_values(data, column):
            score_columns.append(column)
    return score_columns


def detect_risk_indicator_columns(data: pd.DataFrame) -> list[str]:
    """Detect columns that are useful academic risk indicators."""
    risk_keywords = [
        "attendance",
        "absence",
        "missed",
        "study",
        "failure",
        "failures",
        "discipline",
        "engagement",
        "behavior",
        "participation",
    ]
    risk_columns: list[str] = []
    for column in data.columns:
        normalized = normalize_lookup(column)
        if any(keyword in normalized for keyword in risk_keywords):
            risk_columns.append(column)
    return risk_columns


def detect_prediction_feature_mapping(data: pd.DataFrame, score_columns: list[str], attendance_column: str | None) -> dict[str, str]:
    """Suggest prediction input mappings from the uploaded dataset."""
    mapping: dict[str, str] = {}

    study_time_candidates = [
        "studytime",
        "study_time",
        "studyhours",
        "study_hours",
        "hoursstudied",
        "hours_studied",
        "studyhoursweekly",
    ]
    absences_candidates = ["absences", "absence", "classesmissed", "missedclasses", "missed_classes"]
    failures_candidates = ["failures", "pastfailures", "previousfailures", "failedsubjects", "failed_subjects"]

    study_time_column = find_matching_column(data, study_time_candidates)
    if not study_time_column:
        study_time_column = find_column_by_keywords(
            data,
            ["study", "time"],
            exclude_keywords=["attendance", "score", "grade"],
            require_numeric=True,
        )
    if not study_time_column:
        study_time_column = find_column_by_keywords(
            data,
            ["study", "hour"],
            exclude_keywords=["attendance", "score", "grade"],
            require_numeric=True,
        )
    if study_time_column:
        mapping["study_time"] = study_time_column

    absences_column = find_matching_column(data, absences_candidates)
    if absences_column:
        mapping["absences"] = absences_column
    else:
        absence_proxy_column = find_column_by_keywords(
            data,
            ["missed"],
            exclude_keywords=["score", "grade"],
            require_numeric=True,
        )
        if absence_proxy_column:
            mapping["absences"] = absence_proxy_column
        elif attendance_column and has_numeric_values(data, attendance_column):
            mapping["absences"] = attendance_column

    failures_column = find_matching_column(data, failures_candidates)
    if not failures_column:
        failures_column = find_column_by_keywords(
            data,
            ["fail"],
            exclude_keywords=["passfail", "status", "result", "outcome"],
            require_numeric=True,
        )
    if failures_column:
        mapping["failures"] = failures_column

    if score_columns:
        mapping["previous_grade_1"] = score_columns[0]
    if len(score_columns) >= 2:
        mapping["previous_grade_2"] = score_columns[1]
    elif score_columns:
        mapping["previous_grade_2"] = score_columns[0]

    return mapping


def detect_default_feature_columns(
    data: pd.DataFrame,
    target_column: str | None,
    score_columns: list[str],
    risk_indicator_columns: list[str],
) -> list[str]:
    """Suggest general-purpose modeling features for classification."""
    excluded_columns = set(score_columns)
    if target_column:
        excluded_columns.add(target_column)

    preferred_columns = []
    for column in risk_indicator_columns + score_columns:
        if column in data.columns and column not in excluded_columns and column not in preferred_columns:
            preferred_columns.append(column)

    for column in data.columns:
        normalized = normalize_lookup(column)
        if column in excluded_columns:
            continue
        if normalized == "id" or normalized.endswith("id") or "studentid" in normalized:
            continue
        if column not in preferred_columns:
            preferred_columns.append(column)

    return preferred_columns


def detect_default_clustering_features(data: pd.DataFrame, score_columns: list[str], risk_indicator_columns: list[str]) -> list[str]:
    """Suggest numeric clustering features from score and risk signals."""
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    preferred: list[str] = []
    for column in score_columns + risk_indicator_columns:
        if column in numeric_columns and column not in preferred:
            preferred.append(column)
    if preferred:
        return preferred[: min(5, len(preferred))]
    return numeric_columns[: min(5, len(numeric_columns))]


def build_auto_schema_mapping(data: pd.DataFrame) -> SchemaMapping:
    """Auto-detect semantic dataset roles from uploaded academic data."""
    target_column = detect_target_column(data)
    attendance_column = detect_attendance_column(data)
    score_columns = detect_score_columns(data)
    risk_indicator_columns = detect_risk_indicator_columns(data)
    prediction_mapping = detect_prediction_feature_mapping(data, score_columns, attendance_column)

    performance_columns = score_columns[:3] if score_columns else []
    clustering_features = detect_default_clustering_features(data, score_columns, risk_indicator_columns)
    classification_features = detect_default_feature_columns(data, target_column, score_columns[:1], risk_indicator_columns)

    return {
        "student_id_column": get_student_id_column(data),
        "target_column": target_column,
        "attendance_column": attendance_column,
        "score_columns": score_columns,
        "risk_indicator_columns": risk_indicator_columns,
        "prediction_feature_mapping": prediction_mapping,
        "classification_feature_columns": classification_features,
        "clustering_feature_columns": clustering_features,
        "performance_columns": performance_columns if performance_columns else score_columns[:1],
    }


def normalize_schema_mapping(mapping: SchemaMapping, data: pd.DataFrame) -> SchemaMapping:
    """Remove invalid mapped columns and keep only columns present in the dataset."""
    available_columns = set(map(str, data.columns))
    normalized_mapping: SchemaMapping = {}

    for key, value in mapping.items():
        if isinstance(value, list):
            normalized_mapping[key] = [column for column in value if column in available_columns]
        elif isinstance(value, dict):
            normalized_mapping[key] = {
                mapping_key: column
                for mapping_key, column in value.items()
                if column in available_columns
            }
        elif value in available_columns:
            normalized_mapping[key] = value
        else:
            normalized_mapping[key] = None

    return normalized_mapping


def merge_schema_mapping(data: pd.DataFrame, override_mapping: SchemaMapping | None = None) -> SchemaMapping:
    """Combine auto-detected mapping with any user-provided overrides."""
    auto_mapping = build_auto_schema_mapping(data)
    if not override_mapping:
        return auto_mapping

    merged = dict(auto_mapping)
    for key, value in override_mapping.items():
        if value is None:
            continue
        merged[key] = value
    return normalize_schema_mapping(merged, data)


def build_schema_status_table(data: pd.DataFrame, mapping: SchemaMapping) -> pd.DataFrame:
    """Create a readable table showing how semantic roles map to columns."""
    rows = [
        {"Schema Role": "Student ID", "Mapped Column": mapping.get("student_id_column") or "Not mapped"},
        {"Schema Role": "Target Column", "Mapped Column": mapping.get("target_column") or "Derived from score columns"},
        {"Schema Role": "Attendance Column", "Mapped Column": mapping.get("attendance_column") or "Not mapped"},
        {"Schema Role": "Score Columns", "Mapped Column": ", ".join(mapping.get("score_columns", [])) or "Not mapped"},
        {"Schema Role": "Risk Indicators", "Mapped Column": ", ".join(mapping.get("risk_indicator_columns", [])) or "Not mapped"},
        {
            "Schema Role": "Prediction Features",
            "Mapped Column": ", ".join(
                f"{feature}: {column}" for feature, column in mapping.get("prediction_feature_mapping", {}).items()
            )
            or "Not mapped",
        },
    ]
    return pd.DataFrame(rows)


def mapping_supports_prediction(mapping: SchemaMapping) -> bool:
    """Return True when required live prediction features are mapped."""
    feature_mapping = mapping.get("prediction_feature_mapping", {})
    required = {"study_time", "absences", "previous_grade_1", "previous_grade_2"}
    return isinstance(feature_mapping, dict) and required.issubset(set(feature_mapping))


def mapping_supports_classification(mapping: SchemaMapping) -> bool:
    """Return True when the dataset can support Pass/Fail classification."""
    return bool(mapping.get("target_column")) or bool(mapping.get("score_columns"))


def mapping_supports_clustering(mapping: SchemaMapping) -> bool:
    """Return True when clustering features and performance columns are available."""
    return bool(mapping.get("clustering_feature_columns")) and bool(mapping.get("performance_columns"))
