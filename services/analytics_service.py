"""Analytics services for dashboard KPIs, risk summaries, and home snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.classification import FAIL_LABEL, PASS_LABEL, normalize_pass_fail_target
from models.clustering import AVERAGE_LABEL, HIGH_LABEL, RISK_LABEL
from services.dataset_service import (
    get_active_kpi_dataset,
    get_attendance_column,
    get_score_columns,
    get_schema_value,
    has_numeric_values,
    numeric_series,
)
from utilities.dataset_manager import get_classification_run, get_clustering_run
from utilities.schema_mapping import normalize_lookup


@dataclass(frozen=True)
class KpiCard:
    """Display data for a premium KPI card."""

    title: str
    value: str
    note: str
    icon: str
    accent: str = "default"


def scale_percent_value(value: float | None, reference_values: pd.Series | pd.DataFrame | None = None) -> float | None:
    """Convert common academic scales to percentages while keeping 0-100 values unchanged."""
    if value is None or pd.isna(value):
        return None
    max_value = None
    if reference_values is not None:
        stacked_values = reference_values.stack() if isinstance(reference_values, pd.DataFrame) else reference_values
        numeric_values = pd.to_numeric(stacked_values, errors="coerce").dropna()
        if not numeric_values.empty:
            max_value = float(numeric_values.max())
    if max_value is not None and max_value <= 1.5:
        return float(value) * 100
    if max_value is not None and max_value <= 20:
        return float(value) * 5
    if max_value is None and float(value) <= 1.5:
        return float(value) * 100
    return float(value)


def format_percent(value: float | None) -> str:
    """Format a percentage for display."""
    if value is None or pd.isna(value):
        return "--%"
    return f"{value:.1f}%"


def get_student_id_column(data: pd.DataFrame) -> str | None:
    """Find the schema-mapped student identifier column for unique student counting."""
    mapped_column = get_schema_value("student_id_column")
    return mapped_column if mapped_column in data.columns else None


def get_total_students(data: pd.DataFrame) -> int:
    """Count unique students when an ID column exists, otherwise count rows."""
    student_id_column = get_student_id_column(data)
    if student_id_column:
        unique_students = data[student_id_column].dropna().nunique()
        if unique_students:
            return int(unique_students)
    return len(data)


def calculate_average_score(data: pd.DataFrame, score_columns: list[str]) -> float | None:
    """Calculate the overall average score across detected score fields."""
    if not score_columns:
        return None
    score_values = data.loc[:, score_columns].apply(pd.to_numeric, errors="coerce")
    average_score = score_values.stack().mean()
    return scale_percent_value(float(average_score), score_values) if average_score == average_score else None


def calculate_average_attendance(data: pd.DataFrame, attendance_column: str | None) -> float | None:
    """Calculate average attendance from an attendance-like or absence-like column."""
    if not attendance_column:
        return None
    attendance_values = numeric_series(data, attendance_column)
    if normalize_lookup(attendance_column) in {"absences", "absence", "classesmissed"}:
        absence_values = attendance_values.dropna()
        if absence_values.empty:
            return None
        max_absences = max(float(absence_values.max()), 1.0)
        attendance_percent = (1 - (attendance_values / max_absences)).clip(lower=0, upper=1) * 100
        average_attendance = attendance_percent.mean()
        return float(average_attendance) if average_attendance == average_attendance else None

    average_attendance = attendance_values.mean()
    return scale_percent_value(float(average_attendance), attendance_values) if average_attendance == average_attendance else None


def find_existing_pass_fail_target(data: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    """Find and normalize the schema-mapped Pass/Fail target column."""
    target_column = get_schema_value("target_column")
    if target_column in data.columns:
        target = normalize_pass_fail_target(data[target_column])
        valid_target = target.dropna()
        if not valid_target.empty and valid_target.isin([PASS_LABEL, FAIL_LABEL]).all():
            return target, target_column
    return None, None


def derive_pass_fail_target(data: pd.DataFrame, score_columns: list[str]) -> tuple[pd.Series | None, str]:
    """Create a Pass/Fail target from average score when no target column exists."""
    if not score_columns:
        return None, "No Pass/Fail or score columns detected"

    preferred_target_columns = get_schema_value("performance_columns", []) or []
    target_score_columns = [column for column in preferred_target_columns if column in score_columns] or score_columns
    score_values = data.loc[:, target_score_columns].apply(pd.to_numeric, errors="coerce")
    row_average = score_values.mean(axis=1)
    max_score_value = score_values.stack().max()
    if max_score_value == max_score_value and float(max_score_value) <= 1.5:
        threshold = 0.6
    elif max_score_value == max_score_value and float(max_score_value) <= 20:
        threshold = 10.0
    else:
        threshold = 60.0
    target = row_average.ge(threshold).map({True: PASS_LABEL, False: FAIL_LABEL}).astype("string")
    target[row_average.isna()] = pd.NA
    return target, f"Derived from {', '.join(target_score_columns)} at threshold {threshold:g}"


def get_pass_fail_target(data: pd.DataFrame, score_columns: list[str]) -> tuple[pd.Series | None, str | None, str]:
    """Return the best available Pass/Fail labels for KPI calculation."""
    existing_target, target_column = find_existing_pass_fail_target(data)
    if existing_target is not None:
        return existing_target, target_column, f"From `{target_column}`"
    derived_target, target_source = derive_pass_fail_target(data, score_columns)
    return derived_target, None, target_source


def calculate_pass_fail_kpis(target: pd.Series | None) -> dict[str, int | float | None]:
    """Calculate pass/fail rates and at-risk counts from a target series."""
    if target is None:
        return {"pass_rate": None, "fail_rate": None, "pass_count": None, "fail_count": None}

    valid_target = target.dropna()
    if valid_target.empty:
        return {"pass_rate": None, "fail_rate": None, "pass_count": None, "fail_count": None}

    pass_count = int((valid_target == PASS_LABEL).sum())
    fail_count = int((valid_target == FAIL_LABEL).sum())
    total = len(valid_target)
    return {
        "pass_rate": (pass_count / total) * 100,
        "fail_rate": (fail_count / total) * 100,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }


def choose_home_feature_columns(
    data: pd.DataFrame,
    target_column: str | None,
    score_columns: list[str],
) -> list[str]:
    """Choose safe default feature columns for Home prediction accuracy."""
    excluded_columns = set(score_columns)
    if target_column:
        excluded_columns.add(target_column)

    feature_columns: list[str] = []
    for column in data.columns:
        normalized = normalize_lookup(column)
        if column in excluded_columns:
            continue
        if normalized == "id" or normalized.endswith("id") or "studentid" in normalized:
            continue
        if has_numeric_values(data, column) or data[column].dtype.name in {"object", "category", "bool"}:
            feature_columns.append(column)
    return feature_columns


def get_latest_prediction_accuracy() -> tuple[float | None, str | None]:
    """Read the latest Logistic Regression accuracy from the classification page if available."""
    run = get_classification_run()
    if not run or not getattr(run, "results", None):
        return None, None

    logistic_result = next(
        (result for result in run.results if result.model_name.lower() == "logistic regression"),
        run.results[0],
    )
    return float(logistic_result.accuracy) * 100, "Latest trained model"


def calculate_home_prediction_accuracy(
    data: pd.DataFrame,
    target_column: str | None,
    score_columns: list[str],
) -> tuple[float | None, str]:
    """Read prediction accuracy from the latest trained model without retraining on Home render."""
    latest_accuracy, latest_source = get_latest_prediction_accuracy()
    if latest_accuracy is not None:
        return latest_accuracy, latest_source or "Latest trained model"

    if not target_column and not score_columns:
        return None, "Pass/Fail target unavailable"
    return None, "Run or refresh classification to calculate accuracy"


def get_latest_cluster_counts(total_students: int) -> tuple[dict[str, int] | None, str | None]:
    """Return cluster label counts from the latest K-Means run when available."""
    run = get_clustering_run()
    assignments = getattr(run, "assignments", None) if run else None
    if assignments is not None and "Cluster Label" in assignments.columns:
        label_counts = assignments["Cluster Label"].astype("string").value_counts(dropna=True)
        cluster_counts = {
            HIGH_LABEL: int(label_counts.get(HIGH_LABEL, 0)),
            AVERAGE_LABEL: int(label_counts.get(AVERAGE_LABEL, 0)),
            RISK_LABEL: int(label_counts.get(RISK_LABEL, 0)),
        }
        clustered_total = sum(cluster_counts.values())
        if 0 < clustered_total <= total_students:
            return cluster_counts, "From latest K-Means run"
    return None, None


def get_latest_cluster_label_series(total_rows: int) -> tuple[pd.Series | None, str | None]:
    """Align latest K-Means cluster labels back to original dataset row positions."""
    run = get_clustering_run()
    assignments = getattr(run, "assignments", None) if run else None
    if assignments is None or "Cluster Label" not in assignments.columns or "Student Row" not in assignments.columns:
        return None, None

    row_lookup = pd.DataFrame(
        {
            "row_index": pd.to_numeric(assignments["Student Row"], errors="coerce") - 1,
            "label": assignments["Cluster Label"].astype("string"),
        }
    ).dropna(subset=["row_index", "label"])
    if row_lookup.empty:
        return None, None

    row_lookup["row_index"] = row_lookup["row_index"].astype(int)
    row_lookup = row_lookup[row_lookup["row_index"].between(0, max(total_rows - 1, 0))]
    if row_lookup.empty:
        return None, None

    label_series = pd.Series(pd.NA, index=range(total_rows), dtype="string")
    label_series.loc[row_lookup["row_index"]] = row_lookup["label"].tolist()
    return label_series, "From latest K-Means run"


def get_cluster_at_risk_count(default_fail_count: int | None, total_students: int) -> tuple[int | None, str]:
    """Use a clustering run's at-risk label when available, otherwise use failing records."""
    cluster_counts, cluster_source = get_latest_cluster_counts(total_students)
    if cluster_counts is not None:
        return cluster_counts.get(RISK_LABEL, 0), cluster_source or "From latest K-Means run"
    return default_fail_count, "Failing or below-threshold records"


def calculate_row_average_scores(data: pd.DataFrame, score_columns: list[str]) -> pd.Series:
    """Calculate each student's average score using detected score columns."""
    if not score_columns:
        return pd.Series(dtype="float64")
    score_values = data.loc[:, score_columns].apply(pd.to_numeric, errors="coerce")
    row_average = score_values.mean(axis=1)
    max_score = score_values.stack().max()
    if max_score == max_score and float(max_score) <= 1.5:
        row_average = row_average * 100
    elif max_score == max_score and float(max_score) <= 20:
        row_average = row_average * 5
    return row_average


def build_risk_categories(
    target: pd.Series | None,
    row_average_scores: pd.Series,
    attendance_values: pd.Series | None,
) -> pd.Series:
    """Create simple academic risk categories for Home charting."""
    if not row_average_scores.empty:
        category_index = row_average_scores.index
    elif target is not None:
        category_index = target.index
    elif attendance_values is not None:
        category_index = attendance_values.index
    else:
        return pd.Series(dtype="string")

    categories = pd.Series("Average Progress", index=category_index, dtype="string")

    if not row_average_scores.empty:
        categories[row_average_scores >= 75] = "High Performers"
        categories[row_average_scores < 60] = "At-Risk Students"

    if attendance_values is not None and not attendance_values.empty:
        if attendance_values.name and normalize_lookup(attendance_values.name) in {"absences", "absence", "classesmissed"}:
            max_absences = max(float(attendance_values.dropna().max()), 1.0)
            attendance_percent = (1 - (attendance_values / max_absences)).clip(lower=0, upper=1) * 100
        else:
            attendance_percent = attendance_values.map(lambda value: scale_percent_value(value, attendance_values))
        categories[attendance_percent < 75] = "At-Risk Students"
        categories[(attendance_percent >= 75) & (attendance_percent < 88) & (categories != "At-Risk Students")] = (
            "Average Progress"
        )

    if target is not None:
        aligned_target = target.reindex(categories.index)
        categories[aligned_target == FAIL_LABEL] = "At-Risk Students"
        if not row_average_scores.empty:
            categories[(aligned_target == PASS_LABEL) & (row_average_scores >= 75)] = "High Performers"
        else:
            categories[aligned_target == PASS_LABEL] = "High Performers"

    return categories


def build_home_kpis() -> dict[str, object]:
    """Calculate all Home dashboard KPIs from the active student dataset."""
    data, dataset_name, source_label = get_active_kpi_dataset()
    if data.empty:
        return build_empty_home_snapshot(data, dataset_name, source_label)

    score_columns = get_score_columns(data)
    attendance_column = get_attendance_column(data)
    average_score = calculate_average_score(data, score_columns)
    average_attendance = calculate_average_attendance(data, attendance_column)
    average_absences = None
    attendance_metric_title = "Average Attendance"
    attendance_metric_note = attendance_column or "Attendance field not detected"
    if attendance_column and normalize_lookup(attendance_column) in {"absences", "absence", "classesmissed"}:
        average_absences = float(pd.to_numeric(data[attendance_column], errors="coerce").mean())
        attendance_metric_title = "Average Absences"
        attendance_metric_note = attendance_column
    target, target_column, target_source = get_pass_fail_target(data, score_columns)
    pass_fail = calculate_pass_fail_kpis(target)
    row_average_scores = calculate_row_average_scores(data, score_columns)
    attendance_values = numeric_series(data, attendance_column) if attendance_column else None
    risk_categories = build_risk_categories(target, row_average_scores, attendance_values)
    total_students = get_total_students(data)
    cluster_counts, cluster_source = get_latest_cluster_counts(total_students)
    cluster_labels, cluster_label_source = get_latest_cluster_label_series(len(data))
    if cluster_labels is not None and not cluster_labels.dropna().empty:
        risk_categories = cluster_labels
    at_risk_count, at_risk_source = get_cluster_at_risk_count(pass_fail["fail_count"], total_students)
    prediction_accuracy, accuracy_source = calculate_home_prediction_accuracy(data, target_column, score_columns)

    cards = [
        KpiCard("Total Students", f"{total_students:,}", source_label, "ST", "cyan"),
        KpiCard("Pass Rate", format_percent(pass_fail["pass_rate"]), target_source, "PS", "green"),
        KpiCard("Fail Rate", format_percent(pass_fail["fail_rate"]), target_source, "FL", "rose"),
        KpiCard("Average Score", format_percent(average_score), f"{len(score_columns)} score field(s)", "AV", "violet"),
        KpiCard(
            attendance_metric_title,
            "--%"  if average_absences is None and average_attendance is None else f"{average_absences:.1f}" if average_absences is not None else format_percent(average_attendance),
            attendance_metric_note,
            "AT",
            "amber",
        ),
        KpiCard("Prediction Accuracy", format_percent(prediction_accuracy), accuracy_source, "AI", "teal"),
        KpiCard(
            "At-Risk Students",
            "--" if at_risk_count is None else f"{at_risk_count:,}",
            at_risk_source,
            "RK",
            "danger",
        ),
    ]

    return {
        "data": data,
        "dataset_name": dataset_name,
        "source_label": source_label,
        "score_columns": score_columns,
        "attendance_column": attendance_column,
        "average_score": average_score,
        "average_attendance": average_attendance,
        "average_absences": average_absences,
        "attendance_metric_title": attendance_metric_title,
        "row_average_scores": row_average_scores,
        "target": target,
        "risk_categories": risk_categories,
        "pass_rate": pass_fail["pass_rate"],
        "fail_rate": pass_fail["fail_rate"],
        "pass_count": pass_fail["pass_count"],
        "fail_count": pass_fail["fail_count"],
        "prediction_accuracy": prediction_accuracy,
        "at_risk_count": at_risk_count,
        "cluster_counts": cluster_counts,
        "cluster_source": cluster_source or cluster_label_source,
        "target_source": target_source,
        "cards": cards,
    }


def build_empty_home_snapshot(data: pd.DataFrame, dataset_name: str, source_label: str) -> dict[str, object]:
    """Return an empty dashboard state until the user uploads a dataset."""
    cards = [
        KpiCard("Total Students", "0", source_label, "ST", "cyan"),
        KpiCard("Pass Rate", "--%", "Upload a dataset to calculate", "PS", "green"),
        KpiCard("Fail Rate", "--%", "Upload a dataset to calculate", "FL", "rose"),
        KpiCard("Average Score", "--%", "Waiting for score columns", "AV", "violet"),
        KpiCard("Attendance Metric", "--%", "Waiting for attendance or absence fields", "AT", "amber"),
        KpiCard("Prediction Accuracy", "--%", "Upload data to estimate", "AI", "teal"),
        KpiCard("At-Risk Students", "--", "Upload data to calculate", "RK", "danger"),
    ]
    return {
        "data": data,
        "dataset_name": dataset_name,
        "source_label": source_label,
        "score_columns": [],
        "attendance_column": None,
        "average_score": None,
        "average_attendance": None,
        "average_absences": None,
        "attendance_metric_title": "Attendance Metric",
        "row_average_scores": pd.Series(dtype="float64"),
        "target": None,
        "risk_categories": pd.Series(dtype="string"),
        "pass_rate": None,
        "fail_rate": None,
        "pass_count": None,
        "fail_count": None,
        "prediction_accuracy": None,
        "at_risk_count": None,
        "cluster_counts": None,
        "cluster_source": None,
        "target_source": "Upload a student dataset",
        "cards": cards,
    }
