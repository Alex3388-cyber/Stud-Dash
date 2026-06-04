"""AI insights engine for dynamic academic trend analysis.

This module converts dashboard KPI snapshots into readable academic insights.
The engine focuses on:

- attendance impact
- failure risk warnings
- study habit recommendations
- prediction trend summaries
- top-performing student groups

The output is UI-friendly so Streamlit pages can render premium assistant cards
without duplicating the trend logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class AiInsight:
    """Display data for one AI academic insight card."""

    title: str
    label: str
    message: str
    recommendation: str
    icon: str
    severity: str = "info"


def get_group_columns(data: pd.DataFrame, normalize_lookup: Callable[[str], str]) -> list[str]:
    """Find useful categorical columns for group-performance insights."""
    blocked_keywords = {"pass", "fail", "status", "result", "outcome", "id"}
    group_columns: list[str] = []
    for column in data.columns:
        normalized = normalize_lookup(column)
        if any(keyword in normalized for keyword in blocked_keywords):
            continue
        if pd.api.types.is_numeric_dtype(data[column]) or pd.api.types.is_datetime64_any_dtype(data[column]):
            continue
        unique_count = data[column].nunique(dropna=True)
        if 1 < unique_count <= 12:
            group_columns.append(column)
    return group_columns


def get_top_performing_group(
    data: pd.DataFrame,
    row_average_scores: pd.Series,
    normalize_lookup: Callable[[str], str],
) -> tuple[str | None, float | None]:
    """Identify the strongest categorical group by average score."""
    if row_average_scores.empty or row_average_scores.dropna().empty:
        return None, None

    analysis_data = data.copy()
    analysis_data["Average Score"] = row_average_scores
    best_group_label = None
    best_group_score = None

    for column in get_group_columns(data, normalize_lookup):
        grouped_scores = analysis_data.groupby(column, dropna=True)["Average Score"].mean().dropna()
        if grouped_scores.empty:
            continue
        top_value = grouped_scores.idxmax()
        top_score = float(grouped_scores.max())
        if best_group_score is None or top_score > best_group_score:
            best_group_label = f"{column}: {top_value}"
            best_group_score = top_score

    return best_group_label, best_group_score


def build_ai_insights(
    snapshot: dict[str, object],
    *,
    get_total_students: Callable[[pd.DataFrame], int],
    format_percent: Callable[[float | None], str],
    normalize_lookup: Callable[[str], str],
) -> list[AiInsight]:
    """Generate dynamic academic insights from dataset trends."""
    data = snapshot["data"]
    total_students = max(get_total_students(data), 1)
    insights: list[AiInsight] = []

    average_attendance = snapshot.get("average_attendance")
    if isinstance(average_attendance, float):
        if average_attendance < 75:
            insights.append(
                AiInsight(
                    "Low Attendance Warning",
                    "Risk alert",
                    f"Average attendance is {format_percent(average_attendance)}, below the recommended academic support threshold.",
                    "Prioritize attendance monitoring, advisor follow-up, and short weekly check-ins for low-attendance students.",
                    "AT",
                    "critical",
                )
            )
        elif average_attendance < 88:
            insights.append(
                AiInsight(
                    "Attendance Needs Monitoring",
                    "Early signal",
                    f"Average attendance is {format_percent(average_attendance)}, which may weaken performance consistency.",
                    "Review attendance by class group and encourage early intervention before scores decline.",
                    "AT",
                    "warning",
                )
            )
        else:
            insights.append(
                AiInsight(
                    "Attendance Trend Is Strong",
                    "Positive pattern",
                    f"Average attendance is {format_percent(average_attendance)}, supporting stronger academic continuity.",
                    "Maintain engagement strategies and compare attendance against score trends for hidden risk pockets.",
                    "AT",
                    "positive",
                )
            )

    at_risk_count = snapshot.get("at_risk_count")
    if isinstance(at_risk_count, int):
        risk_share = (at_risk_count / total_students) * 100
        severity = "critical" if risk_share >= 30 else "warning" if risk_share >= 15 else "positive"
        insights.append(
            AiInsight(
                "High-Risk Student Alert" if severity != "positive" else "Risk Level Is Controlled",
                "Student risk",
                f"{at_risk_count:,} student(s), about {risk_share:.1f}% of the dataset, are currently flagged as at-risk.",
                "Use clustering and Pass/Fail review together to assign tutoring, attendance support, or advisor follow-up.",
                "RK",
                severity,
            )
        )

    pass_rate = snapshot.get("pass_rate")
    fail_rate = snapshot.get("fail_rate")
    if isinstance(pass_rate, float) and isinstance(fail_rate, float):
        severity = "positive" if pass_rate >= 75 else "warning" if pass_rate >= 60 else "critical"
        insights.append(
            AiInsight(
                "Prediction Trend",
                "Pass/Fail pattern",
                f"Current pass rate is {format_percent(pass_rate)} while fail rate is {format_percent(fail_rate)}.",
                "Track this trend after each upload and compare it with attendance and score distribution shifts.",
                "PF",
                severity,
            )
        )

    row_average_scores = snapshot.get("row_average_scores")
    if isinstance(row_average_scores, pd.Series) and not row_average_scores.dropna().empty:
        low_score_count = int((row_average_scores < 60).sum())
        high_score_count = int((row_average_scores >= 80).sum())
        if low_score_count:
            insights.append(
                AiInsight(
                    "Performance Pattern",
                    "Score distribution",
                    f"{low_score_count:,} student(s) are below the 60% performance threshold.",
                    "Create a remediation list from the low-score segment and review weak assessment areas.",
                    "SC",
                    "warning" if low_score_count / total_students < 0.3 else "critical",
                )
            )
        elif high_score_count:
            insights.append(
                AiInsight(
                    "Strong Performance Pattern",
                    "Score distribution",
                    f"{high_score_count:,} student(s) are scoring at or above 80%.",
                    "Offer enrichment tasks and peer mentoring opportunities to sustain high performance.",
                    "SC",
                    "positive",
                )
            )

        top_group, top_group_score = get_top_performing_group(data, row_average_scores, normalize_lookup)
        if top_group and top_group_score is not None:
            insights.append(
                AiInsight(
                    "Top-Performing Group",
                    "Group insight",
                    f"{top_group} has the strongest average score at {format_percent(top_group_score)}.",
                    "Compare this group with lower-performing groups to identify support practices worth scaling.",
                    "TG",
                    "positive",
                )
            )

    study_time_column = next(
        (column for column in data.columns if normalize_lookup(column) in {"studytime", "studyhours", "study_time", "studyhoursweekly"}),
        None,
    )
    if study_time_column and isinstance(row_average_scores, pd.Series) and not row_average_scores.dropna().empty:
        study_time_values = pd.to_numeric(data[study_time_column], errors="coerce")
        study_analysis = pd.DataFrame(
            {"study_time": study_time_values, "Average Score": row_average_scores}
        ).dropna()
        if not study_analysis.empty:
            low_study_score = study_analysis.loc[study_analysis["study_time"] <= study_analysis["study_time"].median(), "Average Score"].mean()
            high_study_score = study_analysis.loc[study_analysis["study_time"] > study_analysis["study_time"].median(), "Average Score"].mean()
            if low_study_score == low_study_score and high_study_score == high_study_score:
                if high_study_score - low_study_score >= 8:
                    insights.append(
                        AiInsight(
                            "Study Habit Recommendation",
                            "Study pattern",
                            f"Students with stronger study habits are outperforming lower-study peers by about {high_study_score - low_study_score:.1f} points.",
                            "Promote structured weekly study plans and targeted revision support for low-study students.",
                            "SH",
                            "positive",
                        )
                    )
                elif low_study_score < 60:
                    insights.append(
                        AiInsight(
                            "Study Time Risk",
                            "Study pattern",
                            "Lower study-time students are clustering near the at-risk performance threshold.",
                            "Encourage minimum weekly study routines and track whether extra practice improves short-term grades.",
                            "SH",
                            "warning",
                        )
                    )

    failures_column = next(
        (column for column in data.columns if normalize_lookup(column) in {"failures", "pastfailures", "previousfailures"}),
        None,
    )
    if failures_column:
        failure_values = pd.to_numeric(data[failures_column], errors="coerce")
        repeat_failure_count = int((failure_values >= 2).sum())
        if repeat_failure_count:
            severity = "critical" if repeat_failure_count / total_students >= 0.2 else "warning"
            insights.append(
                AiInsight(
                    "Failure Risk Warning",
                    "Academic history",
                    f"{repeat_failure_count:,} student(s) have two or more previous failures, which is a strong academic risk signal.",
                    "Prioritize those students for tutoring, advisor review, and short-cycle progress checks.",
                    "FR",
                    severity,
                )
            )

    cluster_counts = snapshot.get("cluster_counts")
    if isinstance(cluster_counts, dict) and cluster_counts:
        high_cluster_count = int(cluster_counts.get("High Performers", 0))
        average_cluster_count = int(cluster_counts.get("Average Performers", 0))
        if high_cluster_count and average_cluster_count:
            insights.append(
                AiInsight(
                    "Cluster Trend Summary",
                    "Segmentation",
                    f"The latest K-Means run identified {high_cluster_count:,} high-performing students and {average_cluster_count:,} students in the middle academic band.",
                    "Use the middle cluster as the main intervention opportunity before students slide into the at-risk segment.",
                    "KM",
                    "info",
                )
            )

    prediction_accuracy = snapshot.get("prediction_accuracy")
    if isinstance(prediction_accuracy, float):
        severity = "positive" if prediction_accuracy >= 75 else "warning" if prediction_accuracy >= 60 else "critical"
        insights.append(
            AiInsight(
                "Model Reliability Signal",
                "Prediction quality",
                f"Current prediction accuracy is {format_percent(prediction_accuracy)}.",
                "Improve reliability by uploading more balanced Pass/Fail records and retraining after preprocessing.",
                "AI",
                severity,
            )
        )

    if not insights:
        insights.append(
            AiInsight(
                "Upload More Academic Data",
                "AI insight",
                "The dashboard needs score, attendance, or Pass/Fail fields to generate stronger academic intelligence.",
                "Upload a student dataset with attendance, assessment scores, and outcomes for richer insights.",
                "AI",
                "info",
            )
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    return sorted(insights, key=lambda insight: severity_order.get(insight.severity, 2))[:6]
