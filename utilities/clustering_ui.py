"""Streamlit UI for K-Means student segmentation.

The UI lets users select numeric columns, runs K-Means with three clusters, and
shows the resulting student groups with Plotly visualizations, interpretation
cards, and cluster KPI summaries.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from models.clustering import AVERAGE_LABEL, HIGH_LABEL, RISK_LABEL
from preprocessing.exploration_analysis import get_numeric_columns
from services.clustering_service import run_clustering
from utilities.dataset_manager import get_schema_mapping, set_clustering_run
from visualizations.plotly_theme import PREMIUM_COLORWAY, apply_premium_chart_theme


CLUSTER_COLORS = {
    RISK_LABEL: PREMIUM_COLORWAY[4],
    AVERAGE_LABEL: PREMIUM_COLORWAY[2],
    HIGH_LABEL: PREMIUM_COLORWAY[0],
}


def get_default_clustering_features(data: pd.DataFrame, schema_mapping: dict[str, object] | None) -> list[str]:
    """Suggest numeric columns that are useful for student segmentation."""
    schema_mapping = schema_mapping or {}
    numeric_columns = get_numeric_columns(data)
    preferred_columns = [column for column in schema_mapping.get("clustering_feature_columns", []) or [] if column in numeric_columns]
    return preferred_columns[:5] if preferred_columns else numeric_columns[: min(5, len(numeric_columns))]


def get_default_performance_columns(
    data: pd.DataFrame,
    feature_columns: list[str],
    schema_mapping: dict[str, object] | None,
) -> list[str]:
    """Suggest columns used to interpret High, Average, and At-Risk groups."""
    schema_mapping = schema_mapping or {}
    numeric_columns = get_numeric_columns(data)
    performance_columns = [column for column in schema_mapping.get("performance_columns", []) or [] if column in numeric_columns]
    return performance_columns[:3] if performance_columns else feature_columns[:1]


def render_cluster_plot(data: pd.DataFrame) -> None:
    """Render an interactive Plotly scatter plot of the cluster projection."""
    figure = px.scatter(
        data,
        x="Cluster Axis 1",
        y="Cluster Axis 2",
        color="Cluster Label",
        symbol="Cluster Label",
        color_discrete_map=CLUSTER_COLORS,
        hover_data=["Student Row", "K-Means Cluster ID", "Performance Score"],
        title="K-Means Student Clusters",
    )
    figure.update_traces(
        marker={"size": 12},
        hovertemplate=(
            "<b>%{customdata[2]}</b><br>"
            "Student Row: %{customdata[0]}<br>"
            "Cluster ID: %{customdata[1]}<br>"
            "Performance Score: %{customdata[3]:.1f}<br>"
            "Axis 1: %{x:.2f}<br>"
            "Axis 2: %{y:.2f}<extra></extra>"
        ),
    )
    st.plotly_chart(apply_premium_chart_theme(figure, height=560), width="stretch")


def render_cluster_distribution_chart(interpretations: pd.DataFrame) -> None:
    """Render a donut chart showing the share of each academic cluster."""
    if interpretations.empty:
        return

    figure = px.pie(
        interpretations,
        names="Cluster Label",
        values="Students",
        hole=0.6,
        title="Cluster Distribution",
        color="Cluster Label",
        color_discrete_map=CLUSTER_COLORS,
    )
    figure.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.05 if label == RISK_LABEL else 0.02 for label in interpretations["Cluster Label"]],
        hovertemplate="<b>%{label}</b><br>Students: %{value}<br>Share: %{percent}<extra></extra>",
    )
    st.plotly_chart(apply_premium_chart_theme(figure, height=420), width="stretch")


def render_cluster_summary_cards(interpretations: pd.DataFrame) -> None:
    """Render compact KPI cards for High, Average, and At-Risk segments."""
    if interpretations.empty:
        return

    card_classes = {
        HIGH_LABEL: "cluster-card-high",
        AVERAGE_LABEL: "cluster-card-average",
        RISK_LABEL: "cluster-card-risk",
    }

    columns = st.columns(3)
    for index, label in enumerate([HIGH_LABEL, AVERAGE_LABEL, RISK_LABEL]):
        cluster_row = interpretations.loc[interpretations["Cluster Label"] == label]
        if cluster_row.empty:
            columns[index].markdown(
                f"""
                <div class="cluster-summary-card {card_classes[label]}">
                    <span>{label}</span>
                    <strong>0</strong>
                    <small>No clustered records</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue

        row = cluster_row.iloc[0]
        columns[index].markdown(
            f"""
            <div class="cluster-summary-card {card_classes[label]}">
                <span>{label}</span>
                <strong>{int(row["Students"]):,}</strong>
                <small>{row["Share"]} of clustered students | Avg score {float(row["Average Performance Score"]):.1f}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_cluster_interpretations(interpretations: pd.DataFrame) -> None:
    """Display a readable interpretation for every academic cluster label."""
    for _, row in interpretations.iterrows():
        st.markdown(
            f"""
            <div class="interpretation-card">
                <strong>{row["Cluster Label"]}</strong>
                <span>{row["Students"]} students | {row["Share"]} of clustered records | Average score {row["Average Performance Score"]}</span>
                <p>{row["Interpretation"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kmeans_clustering_module(data: pd.DataFrame, dataset_name: str) -> None:
    """Render the complete K-Means clustering workflow."""
    st.markdown(
        f"""
        <div class="module-intro">
            <strong>Active dataset: {dataset_name}</strong>
            <span>Group students into high performers, average performers, and at-risk students.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data.empty:
        st.warning("The selected dataset is empty. Upload a dataset with records before clustering.")
        return

    schema_mapping = get_schema_mapping() or {}
    numeric_columns = get_numeric_columns(data)
    if not numeric_columns:
        st.warning("K-Means clustering requires numeric columns such as scores, attendance, or study time.")
        return

    st.subheader("Clustering Setup")
    st.caption(
        "K-Means uses numeric feature distances, so selected features are imputed for missing values and scaled before clustering."
    )

    default_features = get_default_clustering_features(data, schema_mapping)
    feature_columns = st.multiselect(
        "Numeric features used for K-Means clustering",
        options=numeric_columns,
        default=default_features,
        help="Choose the numeric fields that should define similarity between students.",
    )

    default_performance_columns = get_default_performance_columns(data, feature_columns, schema_mapping)
    performance_columns = st.multiselect(
        "Performance columns used to label clusters",
        options=numeric_columns,
        default=default_performance_columns,
        help="These columns name the clusters after K-Means runs: highest average becomes High Performers, lowest becomes At-Risk Students.",
    )

    if not feature_columns:
        st.warning("Select at least one numeric feature column for K-Means clustering.")

    if not performance_columns:
        st.warning("Select at least one performance column so clusters can be interpreted.")

    left, right = st.columns(2)
    with left:
        st.number_input("Number of clusters", value=3, disabled=True, help="This dashboard uses exactly three academic segments.")
    with right:
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    if not st.button("Run K-Means Clustering", type="primary", width="stretch"):
        st.info("Select numeric features and run K-Means to generate student segments.")
        return

    if not feature_columns:
        st.error("K-Means cannot run because no numeric feature columns were selected.")
        return

    if not performance_columns:
        st.error("K-Means cannot label clusters without at least one performance column.")
        return

    try:
        run = run_clustering(
            data=data,
            feature_columns=feature_columns,
            performance_columns=performance_columns,
            random_state=int(random_state),
        )
    except Exception as error:
        st.error(f"K-Means clustering failed: {error}")
        return

    set_clustering_run(run)
    st.success("K-Means clustering completed successfully.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Students Clustered", f"{run.clustered_rows:,}")
    metric_columns[1].metric("Clusters", "3")
    metric_columns[2].metric("Features Used", f"{len(run.feature_columns):,}")
    metric_columns[3].metric("Inertia", f"{run.inertia:.2f}")

    st.markdown(
        """
        <div class="section-divider">
            <span></span>
            <strong>Cluster KPI Overview</strong>
            <span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_cluster_summary_cards(run.interpretations)

    plot_tab, overview_tab, labels_tab, interpretation_tab, profiles_tab = st.tabs(
        ["Cluster Plot", "Cluster Overview", "Cluster Labels", "Interpretation", "Cluster Profiles"]
    )

    with plot_tab:
        render_cluster_plot(run.visualization_data)

    with overview_tab:
        overview_left, overview_right = st.columns([0.9, 1.1])
        with overview_left:
            render_cluster_distribution_chart(run.interpretations)
        with overview_right:
            st.subheader("Cluster Interpretation Cards")
            render_cluster_interpretations(run.interpretations)

    with labels_tab:
        st.subheader("Student Cluster Labels")
        st.dataframe(run.assignments, width="stretch", hide_index=True)

    with interpretation_tab:
        st.subheader("Cluster Interpretation")
        render_cluster_interpretations(run.interpretations)

    with profiles_tab:
        st.subheader("Cluster Profiles")
        st.dataframe(run.profiles, width="stretch", hide_index=True)
