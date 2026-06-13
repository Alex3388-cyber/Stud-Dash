"""Streamlit data exploration module with Plotly visualizations.

The module reads the active dataframe, lets the user choose columns dynamically,
and renders summary statistics, frequency tables, correlations, and charts.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from preprocessing.exploration_analysis import (
    build_correlation_matrix,
    build_frequency_table,
    build_summary_statistics,
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
)
from visualizations.plotly_theme import PREMIUM_COLORWAY, apply_premium_chart_theme


DEFAULT_COLOR = PREMIUM_COLORWAY[0]
DEFAULT_CHART_HEIGHT = 420


def apply_chart_theme(figure):
    """Apply a consistent dashboard-friendly Plotly theme."""
    return apply_premium_chart_theme(figure, height=DEFAULT_CHART_HEIGHT)


def get_default_column(columns: list[str], keywords: list[str], fallback_index: int = 0) -> str | None:
    """Pick a sensible default column for automatic chart rendering."""
    normalized_columns = {column.lower().strip(): column for column in columns}
    for preferred_column in ["g3", "g2", "g1", "studytime", "absences", "failures"]:
        if preferred_column in normalized_columns and any(keyword in preferred_column for keyword in keywords):
            return normalized_columns[preferred_column]

    for keyword in keywords:
        for column in columns:
            if keyword in column.lower():
                return column
    if columns:
        return columns[min(fallback_index, len(columns) - 1)]
    return None


def render_visual_overview(data: pd.DataFrame) -> None:
    """Render several default charts immediately so graphs are visible on page load."""
    st.subheader("Visual Overview")

    numeric_columns = get_numeric_columns(data)
    categorical_columns = get_categorical_columns(data)

    if not numeric_columns and not categorical_columns:
        st.info("This dataset does not contain columns that can be charted.")
        return

    left, right = st.columns(2)

    with left:
        if categorical_columns:
            category_column = get_default_column(categorical_columns, ["school", "sex", "address", "gender", "course", "lunch", "education"])
            chart_data = build_frequency_table(data, category_column, top_n=10)
            figure = px.bar(
                chart_data,
                x=category_column,
                y="Frequency",
                title=f"Records by {category_column}",
                color_discrete_sequence=[DEFAULT_COLOR],
            )
            st.plotly_chart(apply_chart_theme(figure), width="stretch")
        else:
            st.info("No categorical column available for the default bar chart.")

    with right:
        if numeric_columns:
            score_column = get_default_column(numeric_columns, ["g3", "score", "mark", "grade", "attendance"])
            figure = px.histogram(
                data,
                x=score_column,
                nbins=15,
                title=f"Distribution of {score_column}",
                color_discrete_sequence=[PREMIUM_COLORWAY[2]],
            )
            st.plotly_chart(apply_chart_theme(figure), width="stretch")
        else:
            st.info("No numeric column available for the default histogram.")

    lower_left, lower_right = st.columns(2)

    with lower_left:
        if len(numeric_columns) >= 2:
            x_column = get_default_column(numeric_columns, ["studytime", "study", "absences", "attendance"], fallback_index=0)
            y_column = get_default_column(
                [column for column in numeric_columns if column != x_column],
                ["g3", "score", "mark", "grade"],
                fallback_index=0,
            )
            figure = px.scatter(
                data,
                x=x_column,
                y=y_column,
                color=categorical_columns[0] if categorical_columns else None,
                title=f"{y_column} vs {x_column}",
            )
            st.plotly_chart(apply_chart_theme(figure), width="stretch")
        else:
            st.info("At least two numeric columns are needed for the default scatter plot.")

    with lower_right:
        if len(numeric_columns) >= 2:
            correlation_columns = numeric_columns[: min(5, len(numeric_columns))]
            correlation_matrix = build_correlation_matrix(data, correlation_columns)
            figure = px.imshow(
                correlation_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Tealrose",
                title="Numeric Correlation Overview",
            )
            st.plotly_chart(apply_chart_theme(figure), width="stretch")
        elif categorical_columns:
            category_column = categorical_columns[0]
            chart_data = build_frequency_table(data, category_column, top_n=8)
            figure = px.pie(
                chart_data,
                names=category_column,
                values="Frequency",
                title=f"Composition by {category_column}",
                hole=0.35,
            )
            st.plotly_chart(apply_chart_theme(figure), width="stretch")


def render_exploration_metrics(data: pd.DataFrame) -> None:
    """Show quick dataset health metrics before deeper exploration."""
    numeric_count = len(get_numeric_columns(data))
    categorical_count = len(get_categorical_columns(data))
    missing_values = int(data.isna().sum().sum())

    columns = st.columns(4)
    columns[0].metric("Rows", f"{len(data):,}")
    columns[1].metric("Columns", f"{len(data.columns):,}")
    columns[2].metric("Numeric Columns", f"{numeric_count:,}")
    columns[3].metric("Missing Values", f"{missing_values:,}")


def render_summary_statistics(data: pd.DataFrame) -> None:
    """Render dynamic summary statistics for user-selected columns."""
    st.subheader("Summary Statistics")

    selected_columns = st.multiselect(
        "Select columns for summary statistics",
        options=list(data.columns),
        default=list(data.columns),
        key="summary_columns",
    )

    if not selected_columns:
        st.warning("Select at least one column to generate summary statistics.")
        return

    # Pandas describe(include='all') summarizes numeric and categorical fields together.
    summary = build_summary_statistics(data, selected_columns)
    st.dataframe(summary, width="stretch", hide_index=True)


def render_frequency_tables(data: pd.DataFrame) -> None:
    """Render frequency tables for categorical or selected columns."""
    st.subheader("Frequency Tables")

    categorical_columns = get_categorical_columns(data)
    column_options = categorical_columns if categorical_columns else list(data.columns)
    if not column_options:
        st.info("No columns are available for frequency analysis.")
        return

    selected_column = st.selectbox(
        "Select a column for frequency analysis",
        options=column_options,
        key="frequency_column",
    )
    top_n = st.slider("Number of categories to show", min_value=5, max_value=50, value=15, step=5)

    # Frequency tables help identify dominant values, rare groups, and data entry issues.
    frequency_table = build_frequency_table(data, selected_column, top_n)
    st.dataframe(frequency_table, width="stretch", hide_index=True)


def render_correlation_analysis(data: pd.DataFrame) -> None:
    """Render correlation controls and a Plotly heatmap for numeric columns."""
    st.subheader("Correlation Analysis")

    numeric_columns = get_numeric_columns(data)
    if len(numeric_columns) < 2:
        st.info("Correlation analysis requires at least two numeric columns.")
        return

    selected_columns = st.multiselect(
        "Select numeric columns",
        options=numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
        key="correlation_columns",
    )
    method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])

    correlation_matrix = build_correlation_matrix(data, selected_columns, method)
    if correlation_matrix.empty:
        st.warning("Select at least two numeric columns to calculate correlations.")
        return

    st.dataframe(correlation_matrix.round(3), width="stretch")

    # Plotly imshow gives an interactive heatmap with hoverable correlation values.
    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Tealrose",
        title="Correlation Heatmap",
    )
    st.plotly_chart(apply_premium_chart_theme(figure, height=460), width="stretch")


def render_bar_chart(data: pd.DataFrame) -> None:
    """Render a dynamic Plotly bar chart."""
    categorical_columns = get_categorical_columns(data)
    numeric_columns = get_numeric_columns(data)

    if not categorical_columns:
        st.info("A bar chart needs at least one categorical column.")
        return

    left, right = st.columns(2)
    with left:
        x_column = st.selectbox("Bar category", categorical_columns, key="bar_x")
    with right:
        y_options = ["Record count"] + numeric_columns
        y_column = st.selectbox("Bar value", y_options, key="bar_y")

    if y_column == "Record count":
        chart_data = build_frequency_table(data, x_column, top_n=25)
        figure = px.bar(
            chart_data,
            x=x_column,
            y="Frequency",
            title=f"Record Count by {x_column}",
            color_discrete_sequence=[DEFAULT_COLOR],
        )
    else:
        # Average numeric values by category so the chart stays readable.
        chart_data = data.groupby(x_column, as_index=False)[y_column].mean()
        figure = px.bar(
            chart_data,
            x=x_column,
            y=y_column,
            title=f"Average {y_column} by {x_column}",
            color_discrete_sequence=[DEFAULT_COLOR],
        )

    st.plotly_chart(apply_chart_theme(figure), width="stretch")


def render_histogram(data: pd.DataFrame) -> None:
    """Render a dynamic Plotly histogram for numeric distributions."""
    numeric_columns = get_numeric_columns(data)
    categorical_columns = get_categorical_columns(data)

    if not numeric_columns:
        st.info("A histogram requires at least one numeric column.")
        return

    left, right = st.columns(2)
    with left:
        x_column = st.selectbox("Histogram column", numeric_columns, key="histogram_x")
    with right:
        color_options = ["None"] + categorical_columns
        color_column = st.selectbox("Group by", color_options, key="histogram_color")

    figure = px.histogram(
        data,
        x=x_column,
        color=None if color_column == "None" else color_column,
        nbins=20,
        title=f"Distribution of {x_column}",
    )
    st.plotly_chart(apply_chart_theme(figure), width="stretch")


def render_scatter_plot(data: pd.DataFrame) -> None:
    """Render a dynamic Plotly scatter plot for relationships between numeric columns."""
    numeric_columns = get_numeric_columns(data)
    categorical_columns = get_categorical_columns(data)

    if len(numeric_columns) < 2:
        st.info("A scatter plot requires at least two numeric columns.")
        return

    left, middle, right = st.columns(3)
    with left:
        x_column = st.selectbox("X-axis", numeric_columns, key="scatter_x")
    with middle:
        default_y_index = 1 if len(numeric_columns) > 1 else 0
        y_column = st.selectbox("Y-axis", numeric_columns, index=default_y_index, key="scatter_y")
    with right:
        color_options = ["None"] + categorical_columns
        color_column = st.selectbox("Color by", color_options, key="scatter_color")

    figure = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=None if color_column == "None" else color_column,
        title=f"{y_column} vs {x_column}",
    )
    st.plotly_chart(apply_chart_theme(figure), width="stretch")


def render_pie_chart(data: pd.DataFrame) -> None:
    """Render a dynamic Plotly pie chart for categorical composition."""
    categorical_columns = get_categorical_columns(data)

    if not categorical_columns:
        st.info("A pie chart requires at least one categorical column.")
        return

    selected_column = st.selectbox("Pie category", categorical_columns, key="pie_column")
    top_n = st.slider("Pie categories to show", min_value=3, max_value=15, value=8, key="pie_top_n")
    chart_data = build_frequency_table(data, selected_column, top_n=top_n)

    figure = px.pie(
        chart_data,
        names=selected_column,
        values="Frequency",
        title=f"Composition by {selected_column}",
        hole=0.35,
    )
    st.plotly_chart(apply_chart_theme(figure), width="stretch")


def render_line_chart(data: pd.DataFrame) -> None:
    """Render a dynamic Plotly line chart for ordered or time-based trends."""
    numeric_columns = get_numeric_columns(data)
    datetime_columns = get_datetime_columns(data)
    x_options = datetime_columns + list(data.columns)
    if not x_options:
        st.info("A line chart requires at least one column for the X-axis.")
        return

    if not numeric_columns:
        st.info("A line chart requires at least one numeric column.")
        return

    left, right = st.columns(2)
    with left:
        x_column = st.selectbox("Line X-axis", x_options, key="line_x")
    with right:
        y_column = st.selectbox("Line Y-axis", numeric_columns, key="line_y")

    # Sort by the X-axis so line charts follow the selected column's natural order.
    chart_data = data[[x_column, y_column]].dropna().sort_values(x_column)
    figure = px.line(
        chart_data,
        x=x_column,
        y=y_column,
        markers=True,
        title=f"{y_column} Trend by {x_column}",
    )
    st.plotly_chart(apply_chart_theme(figure), width="stretch")


def render_interactive_charts(data: pd.DataFrame) -> None:
    """Render all requested Plotly chart types behind a compact chart selector."""
    st.subheader("Interactive Plotly Charts")

    chart_type = st.selectbox(
        "Choose chart type",
        ["Bar Chart", "Histogram", "Scatter Plot", "Pie Chart", "Line Chart"],
    )

    if chart_type == "Bar Chart":
        render_bar_chart(data)
    elif chart_type == "Histogram":
        render_histogram(data)
    elif chart_type == "Scatter Plot":
        render_scatter_plot(data)
    elif chart_type == "Pie Chart":
        render_pie_chart(data)
    else:
        render_line_chart(data)


def render_data_exploration_module(data: pd.DataFrame, dataset_name: str) -> None:
    """Render the complete data exploration workflow."""
    if data.empty:
        st.warning("The selected dataset is empty. Upload a dataset with rows before exploring it.")
        return

    st.markdown(
        f"""
        <div class="module-intro">
            <strong>Active dataset: {dataset_name}</strong>
            <span>Select columns dynamically to inspect patterns, relationships, and distributions.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_exploration_metrics(data)

    tabs = st.tabs(
        [
            "Visual Overview",
            "Summary Statistics",
            "Frequency Tables",
            "Correlation Analysis",
            "Interactive Charts",
            "Data Preview",
        ]
    )

    with tabs[0]:
        render_visual_overview(data)
    with tabs[1]:
        render_summary_statistics(data)
    with tabs[2]:
        render_frequency_tables(data)
    with tabs[3]:
        render_correlation_analysis(data)
    with tabs[4]:
        render_interactive_charts(data)
    with tabs[5]:
        st.subheader("Data Preview")
        st.dataframe(data.head(100), width="stretch")
