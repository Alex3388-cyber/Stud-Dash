"""Plotly chart builders for student performance analysis."""

import pandas as pd
import plotly.express as px

from utilities.constants import SCORE_COLUMNS
from visualizations.plotly_theme import apply_premium_chart_theme


def build_average_score_chart(data: pd.DataFrame):
    """Create a bar chart of average scores by parental education level."""
    chart_data = data.copy()
    chart_data["average_score"] = chart_data[SCORE_COLUMNS].mean(axis=1)
    grouped = (
        chart_data.groupby("parental_education", as_index=False)["average_score"]
        .mean()
        .sort_values("average_score", ascending=False)
    )

    figure = px.bar(
        grouped,
        x="parental_education",
        y="average_score",
        title="Average Score by Parental Education",
        labels={"parental_education": "Parental Education", "average_score": "Average Score"},
    )
    return apply_premium_chart_theme(figure)


def build_study_time_chart(data: pd.DataFrame):
    """Create a scatter chart comparing study time and average score."""
    chart_data = data.copy()
    chart_data["average_score"] = chart_data[SCORE_COLUMNS].mean(axis=1)

    figure = px.scatter(
        chart_data,
        x="study_time_hours",
        y="average_score",
        color="test_preparation_course",
        title="Study Time vs Average Score",
        labels={
            "study_time_hours": "Study Time Hours",
            "average_score": "Average Score",
            "test_preparation_course": "Test Preparation",
        },
    )
    return apply_premium_chart_theme(figure)


def build_score_distribution_chart(data: pd.DataFrame):
    """Create a box plot showing score distributions by subject."""
    melted = data.melt(
        id_vars=["student_id"],
        value_vars=SCORE_COLUMNS,
        var_name="subject",
        value_name="score",
    )

    figure = px.box(
        melted,
        x="subject",
        y="score",
        title="Score Distribution by Subject",
        labels={"subject": "Subject", "score": "Score"},
    )
    return apply_premium_chart_theme(figure)
