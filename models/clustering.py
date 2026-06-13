"""K-Means clustering logic for student performance segmentation.

This module groups students into exactly three clusters:

- High Performers
- Average Performers
- At-Risk Students

K-Means produces numeric cluster IDs such as 0, 1, and 2. Those IDs do not have
academic meaning by themselves, so this module interprets each cluster by
comparing its average performance score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


CLUSTER_COUNT = 3
HIGH_LABEL = "High Performers"
AVERAGE_LABEL = "Average Performers"
RISK_LABEL = "At-Risk Students"
CLUSTER_ORDER = [RISK_LABEL, AVERAGE_LABEL, HIGH_LABEL]


@dataclass(frozen=True)
class ClusteringRun:
    """Complete output from one K-Means clustering run."""

    feature_columns: list[str]
    performance_columns: list[str]
    clustered_rows: int
    inertia: float
    assignments: pd.DataFrame
    profiles: pd.DataFrame
    interpretations: pd.DataFrame
    visualization_data: pd.DataFrame


def remove_all_missing_feature_columns(data: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Remove selected feature columns that contain only missing values."""
    return [column for column in feature_columns if not data[column].isna().all()]


def prepare_numeric_features(data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Impute and scale numeric features before K-Means clustering."""
    valid_feature_columns = remove_all_missing_feature_columns(data, feature_columns)
    if not valid_feature_columns:
        raise ValueError("Select at least one numeric feature with usable values.")

    selected_data = data.loc[:, valid_feature_columns].copy()

    # Rows where every selected clustering feature is missing cannot be placed
    # reliably, so they are removed before fitting K-Means.
    usable_row_mask = ~selected_data.isna().all(axis=1)
    selected_data = selected_data.loc[usable_row_mask]
    if len(selected_data) < CLUSTER_COUNT:
        raise ValueError("K-Means requires at least three usable student records.")

    # K-Means cannot read missing values. Median imputation fills numeric gaps
    # while staying robust to unusually high or low academic values.
    imputer = SimpleImputer(strategy="median")
    imputed_features = imputer.fit_transform(selected_data)

    # K-Means is distance-based. Scaling prevents larger numeric ranges, such as
    # total marks, from overpowering smaller ranges, such as attendance ratios.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    unique_rows = np.unique(scaled_features, axis=0)
    if len(unique_rows) < CLUSTER_COUNT:
        raise ValueError("K-Means requires at least three distinct feature patterns.")

    return selected_data, scaled_features


def build_visualization_coordinates(scaled_features: np.ndarray) -> pd.DataFrame:
    """Create two-dimensional coordinates for an interactive cluster plot."""
    if scaled_features.shape[1] >= 2:
        # PCA compresses all selected features into two axes for visualization.
        # Clustering still uses the full selected feature set, not only these axes.
        coordinates = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
        x_values = coordinates[:, 0]
        y_values = coordinates[:, 1]
    else:
        # With one selected feature, use that feature as the horizontal axis and
        # keep the vertical axis at zero so points can still be displayed.
        x_values = scaled_features[:, 0]
        y_values = np.zeros(len(scaled_features))

    return pd.DataFrame({"Cluster Axis 1": x_values, "Cluster Axis 2": y_values})


def calculate_performance_scores(data: pd.DataFrame, performance_columns: list[str]) -> pd.Series:
    """Calculate each student's average performance score for cluster labeling."""
    if not performance_columns:
        raise ValueError("Select at least one numeric performance column for cluster interpretation.")

    # The performance score is used only to name and interpret clusters. K-Means
    # labels are assigned by distance patterns in the selected clustering features.
    performance_values = data.loc[:, performance_columns].apply(pd.to_numeric, errors="coerce")
    performance_scores = performance_values.mean(axis=1)
    max_score = performance_values.stack().max()
    if max_score == max_score and float(max_score) <= 20:
        performance_scores = performance_scores * 5
    elif max_score == max_score and float(max_score) <= 1.5:
        performance_scores = performance_scores * 100
    return performance_scores


def label_clusters_by_performance(raw_clusters: np.ndarray, performance_scores: pd.Series) -> dict[int, str]:
    """Map raw K-Means IDs to academic labels based on average performance."""
    cluster_scores = (
        pd.DataFrame({"Cluster ID": raw_clusters, "Performance Score": performance_scores.values})
        .groupby("Cluster ID", as_index=False)["Performance Score"]
        .mean()
        .sort_values("Performance Score")
    )

    ordered_labels = [RISK_LABEL, AVERAGE_LABEL, HIGH_LABEL]
    return {
        int(cluster_id): ordered_labels[position]
        for position, cluster_id in enumerate(cluster_scores["Cluster ID"].tolist())
    }


def build_cluster_interpretations(assignments: pd.DataFrame) -> pd.DataFrame:
    """Create human-readable interpretations for each labeled cluster."""
    grouped = assignments.groupby("Cluster Label", observed=False)
    total_students = len(assignments)

    interpretation_text = {
        HIGH_LABEL: (
            "Students in this cluster show the strongest average performance. "
            "They may benefit from enrichment, advanced tasks, and leadership opportunities."
        ),
        AVERAGE_LABEL: (
            "Students in this cluster sit near the middle performance band. "
            "They may benefit from steady monitoring and targeted academic reinforcement."
        ),
        RISK_LABEL: (
            "Students in this cluster show the lowest average performance. "
            "They should be prioritized for early intervention, tutoring, or advisor follow-up."
        ),
    }

    rows = []
    for label in CLUSTER_ORDER:
        if label not in grouped.groups:
            continue

        cluster_data = grouped.get_group(label)
        rows.append(
            {
                "Cluster Label": label,
                "Students": len(cluster_data),
                "Share": f"{(len(cluster_data) / total_students) * 100:.1f}%",
                "Average Performance Score": round(cluster_data["Performance Score"].mean(), 2),
                "Interpretation": interpretation_text[label],
            }
        )

    return pd.DataFrame(rows)


def build_cluster_profiles(
    assignments: pd.DataFrame,
    feature_columns: list[str],
    performance_columns: list[str],
) -> pd.DataFrame:
    """Summarize each cluster using average feature and performance values."""
    profile_columns = list(dict.fromkeys(feature_columns + performance_columns))
    profile_columns = [column for column in profile_columns if column in assignments.columns]

    counts = assignments.groupby("Cluster Label", observed=False).size().rename("Students")
    means = assignments.groupby("Cluster Label", observed=False)[profile_columns].mean().round(2)
    profiles = pd.concat([counts, means], axis=1).reset_index()

    # Keep the table in a meaningful academic order instead of raw K-Means order.
    profiles["Cluster Label"] = pd.Categorical(profiles["Cluster Label"], categories=CLUSTER_ORDER, ordered=True)
    return profiles.sort_values("Cluster Label").reset_index(drop=True)


def run_kmeans_clustering(
    data: pd.DataFrame,
    feature_columns: list[str],
    performance_columns: list[str],
    random_state: int = 42,
) -> ClusteringRun:
    """Cluster students into High, Average, and At-Risk groups with K-Means."""
    selected_features, scaled_features = prepare_numeric_features(data, feature_columns)

    # K-Means separates students into three groups by minimizing the distance
    # between each student and the center of the cluster they are assigned to.
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=random_state, n_init=10)
    raw_clusters = kmeans.fit_predict(scaled_features)

    performance_scores = calculate_performance_scores(data.loc[selected_features.index], performance_columns)
    cluster_label_lookup = label_clusters_by_performance(raw_clusters, performance_scores)
    readable_labels = [cluster_label_lookup[int(cluster_id)] for cluster_id in raw_clusters]

    assignments = data.loc[selected_features.index].copy()
    assignments.insert(0, "Student Row", selected_features.index + 1)
    assignments["K-Means Cluster ID"] = raw_clusters
    assignments["Cluster Label"] = pd.Categorical(readable_labels, categories=CLUSTER_ORDER, ordered=True)
    assignments["Performance Score"] = performance_scores.values
    assignments = assignments.sort_values(["Cluster Label", "Performance Score"], ascending=[True, False])

    visualization_data = build_visualization_coordinates(scaled_features)
    visualization_data["Student Row"] = selected_features.index + 1
    visualization_data["K-Means Cluster ID"] = raw_clusters
    visualization_data["Cluster Label"] = pd.Categorical(readable_labels, categories=CLUSTER_ORDER, ordered=True)
    visualization_data["Performance Score"] = performance_scores.values

    profiles = build_cluster_profiles(assignments, selected_features.columns.tolist(), performance_columns)
    interpretations = build_cluster_interpretations(assignments)

    return ClusteringRun(
        feature_columns=selected_features.columns.tolist(),
        performance_columns=performance_columns,
        clustered_rows=len(assignments),
        inertia=float(kmeans.inertia_),
        assignments=assignments,
        profiles=profiles,
        interpretations=interpretations,
        visualization_data=visualization_data,
    )
