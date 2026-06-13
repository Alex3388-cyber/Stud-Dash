"""Streamlit UI for professional Pass/Fail classification analytics.

The UI collects target settings, feature columns, and train/test options, then
delegates training and evaluation to models/classification.py. It now renders
cross-validation summaries, calibrated probability diagnostics, and feature
importance views in addition to holdout metrics.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from models.classification import (
    PASS_FAIL_TARGET,
    find_pass_fail_column,
)
from preprocessing.exploration_analysis import get_numeric_columns
from services.ml_service import build_model_metrics_table, train_models
from utilities.dataset_manager import get_schema_mapping, set_classification_run
from utilities.trust_ui import (
    build_feature_importance_overview,
    render_dataset_source_banner,
    render_model_explanation_card,
    render_preprocessing_audit_card,
)
from visualizations.plotly_theme import PREMIUM_COLORWAY, PREMIUM_CONTINUOUS_SCALE, apply_premium_chart_theme


def get_default_score_columns(data: pd.DataFrame, schema_mapping: dict[str, object] | None) -> list[str]:
    """Suggest score columns from the active schema mapping."""
    schema_mapping = schema_mapping or {}
    numeric_columns = get_numeric_columns(data)
    mapped_scores = [column for column in schema_mapping.get("score_columns", []) or [] if column in numeric_columns]
    return mapped_scores[:3] if mapped_scores else numeric_columns[:1]


def get_default_feature_columns(
    data: pd.DataFrame,
    schema_mapping: dict[str, object] | None,
    excluded_columns: set[str],
) -> list[str]:
    """Choose classification features from the schema mapping first."""
    schema_mapping = schema_mapping or {}
    mapped_features = [
        column for column in schema_mapping.get("classification_feature_columns", []) or [] if column in data.columns
    ]
    mapped_features = [column for column in mapped_features if column not in excluded_columns]
    if mapped_features:
        return mapped_features

    default_features: list[str] = []
    for column in data.columns:
        lower_column = column.lower()
        if column in excluded_columns:
            continue
        if lower_column.endswith("id") or lower_column == "id" or "student_id" in lower_column:
            continue
        default_features.append(column)
    return default_features


def render_confusion_matrix(result) -> None:
    """Display one model's confusion matrix as both a table and heatmap."""
    left, right = st.columns([0.95, 1.05])
    with left:
        st.dataframe(result.confusion_matrix, width="stretch")
    with right:
        figure = px.imshow(
            result.confusion_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=PREMIUM_CONTINUOUS_SCALE,
            title=f"{result.model_name} Confusion Matrix",
        )
        st.plotly_chart(apply_premium_chart_theme(figure, height=360), width="stretch")


def build_roc_figure(result) -> go.Figure:
    """Render a ROC curve for one model."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.roc_curve["False Positive Rate"],
            y=result.roc_curve["True Positive Rate"],
            mode="lines",
            name=result.model_name,
            line={"color": PREMIUM_COLORWAY[0], "width": 3},
            fill="tozeroy",
            fillcolor="rgba(54,230,194,0.12)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline",
            line={"color": "rgba(168,186,208,0.6)", "width": 1.5, "dash": "dash"},
        )
    )
    figure.update_layout(title=f"{result.model_name} ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    figure.update_yaxes(range=[0, 1.02])
    figure.update_xaxes(range=[0, 1.02])
    return apply_premium_chart_theme(figure, height=360)


def build_precision_recall_figure(result) -> go.Figure:
    """Render a precision-recall curve for one model."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.precision_recall_curve["Recall"],
            y=result.precision_recall_curve["Precision"],
            mode="lines",
            name=result.model_name,
            line={"color": PREMIUM_COLORWAY[2], "width": 3},
            fill="tozeroy",
            fillcolor="rgba(255,209,102,0.12)",
        )
    )
    figure.update_layout(title=f"{result.model_name} Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    figure.update_yaxes(range=[0, 1.02])
    figure.update_xaxes(range=[0, 1.02])
    return apply_premium_chart_theme(figure, height=360)


def build_calibration_figure(result) -> go.Figure:
    """Render a calibration curve for one model."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.calibration_curve["Mean Predicted Probability"],
            y=result.calibration_curve["Observed Pass Rate"],
            mode="lines+markers",
            name=result.model_name,
            line={"color": PREMIUM_COLORWAY[1], "width": 3},
            marker={"size": 10},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ideal",
            line={"color": "rgba(168,186,208,0.6)", "width": 1.5, "dash": "dash"},
        )
    )
    figure.update_layout(
        title=f"{result.model_name} Calibration Curve",
        xaxis_title="Mean Predicted Pass Probability",
        yaxis_title="Observed Pass Rate",
    )
    figure.update_yaxes(range=[0, 1.02])
    figure.update_xaxes(range=[0, 1.02])
    return apply_premium_chart_theme(figure, height=360)


def build_feature_importance_figure(result) -> go.Figure:
    """Render the top holdout permutation importances for one model."""
    importance_table = result.feature_importance.head(12).iloc[::-1]
    figure = px.bar(
        importance_table,
        x="Importance",
        y="Feature",
        orientation="h",
        error_x="Importance Std",
        color="Importance",
        color_continuous_scale=PREMIUM_CONTINUOUS_SCALE,
        title=f"{result.model_name} Feature Importance",
    )
    figure.update_layout(showlegend=False, coloraxis_showscale=False)
    return apply_premium_chart_theme(figure, height=420)


def render_model_evaluation(result) -> None:
    """Render the detailed evaluation panels for one model."""
    st.subheader(result.model_name)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Holdout Accuracy", f"{result.accuracy * 100:.1f}%")
    metric_columns[1].metric("Holdout F1-score", f"{result.f1_score * 100:.1f}%")
    metric_columns[2].metric("ROC-AUC", f"{(result.roc_auc or 0) * 100:.1f}%")
    metric_columns[3].metric("Brier Score", f"{result.brier_score:.3f}" if result.brier_score is not None else "N/A")

    st.caption(result.summary)
    if result.calibration_note:
        st.caption(result.calibration_note)
    if result.cross_validation_note:
        st.caption(result.cross_validation_note)

    curve_left, curve_right = st.columns(2)
    with curve_left:
        st.plotly_chart(build_roc_figure(result), width="stretch")
    with curve_right:
        st.plotly_chart(build_precision_recall_figure(result), width="stretch")

    calibration_left, calibration_right = st.columns([1.0, 1.0])
    with calibration_left:
        st.plotly_chart(build_calibration_figure(result), width="stretch")
    with calibration_right:
        if result.feature_importance.empty:
            st.info("Feature importance could not be estimated for this run.")
        else:
            st.plotly_chart(build_feature_importance_figure(result), width="stretch")

    cv_left, cv_right = st.columns([1.05, 0.95])
    with cv_left:
        st.write("Cross-validation summary")
        if result.cross_validation_metrics.empty:
            st.info("Cross-validation was not available for this dataset size.")
        else:
            cv_table = result.cross_validation_metrics.copy()
            cv_table["Mean"] = cv_table["Mean"].map(lambda value: round(float(value), 4))
            cv_table["Std"] = cv_table["Std"].map(lambda value: round(float(value), 4))
            st.dataframe(cv_table, width="stretch", hide_index=True)
    with cv_right:
        render_confusion_matrix(result)

    if not result.feature_importance.empty:
        st.write("Permutation feature importance")
        display_table = result.feature_importance.copy()
        display_table["Importance"] = display_table["Importance"].map(lambda value: round(float(value), 4))
        display_table["Importance Std"] = display_table["Importance Std"].map(lambda value: round(float(value), 4))
        st.dataframe(display_table.head(15), width="stretch", hide_index=True)


def render_classification_module(data: pd.DataFrame, dataset_name: str) -> None:
    """Render the complete Pass/Fail classification workflow."""
    render_dataset_source_banner()
    st.markdown(
        f"""
        <div class="module-intro">
            <strong>Active dataset: {dataset_name}</strong>
            <span>Train calibrated Logistic Regression, Decision Tree, and Random Forest classifiers for Pass/Fail outcomes.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data.empty:
        st.warning("The selected dataset is empty. Upload a dataset with records before training classifiers.")
        return

    schema_mapping = get_schema_mapping() or {}
    pass_fail_column = schema_mapping.get("target_column") or find_pass_fail_column(data)
    target_mode = "Use existing Pass/Fail column" if pass_fail_column else "Derive Pass/Fail from score columns"

    st.subheader("Target and Feature Setup")
    target_left, target_right = st.columns(2)

    with target_left:
        target_mode = st.radio(
            "Target source",
            ["Use existing Pass/Fail column", "Derive Pass/Fail from score columns"],
            index=0 if pass_fail_column else 1,
            disabled=pass_fail_column is None,
            help="If your dataset has a mapped Pass/Fail column, use it directly. Otherwise derive it from mapped score columns.",
        )

    target_column = pass_fail_column if target_mode == "Use existing Pass/Fail column" else None
    score_columns: list[str] = []
    pass_threshold = 60.0

    with target_right:
        if target_column:
            st.info(f"Using target column: `{target_column}`")
        else:
            numeric_columns = get_numeric_columns(data)
            default_scores = get_default_score_columns(data, schema_mapping)
            score_columns = st.multiselect(
                "Score columns used to create Pass/Fail",
                options=numeric_columns,
                default=default_scores,
                help="These mapped score columns will be used to derive the Pass/Fail target when no direct target column exists.",
            )
            max_score = (
                data.loc[:, score_columns].apply(pd.to_numeric, errors="coerce").stack().max() if score_columns else 100
            )
            if max_score == max_score and float(max_score) <= 20:
                pass_threshold = st.slider("Pass threshold", min_value=0.0, max_value=20.0, value=10.0, step=1.0)
            else:
                pass_threshold = st.slider("Pass threshold", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

    excluded_columns = set(score_columns)
    if target_column:
        excluded_columns.add(target_column)
    default_features = get_default_feature_columns(data, schema_mapping, excluded_columns)
    feature_columns = st.multiselect(
        "Feature columns for training",
        options=list(data.columns),
        default=default_features,
        help="These columns are used to predict Pass/Fail. Any columns that directly define the target are automatically removed to prevent data leakage.",
    )

    if not target_column and not score_columns:
        st.warning("Select at least one score column to derive the Pass/Fail target.")

    if not feature_columns:
        st.warning("Select at least one feature column before training the classifiers.")

    split_left, split_middle, split_right = st.columns(3)
    with split_left:
        test_size_percent = st.slider("Testing data size", min_value=20, max_value=40, value=30, step=5)
    with split_middle:
        cv_folds = st.slider("Cross-validation folds", min_value=3, max_value=7, value=5, step=1)
    with split_right:
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    st.caption(
        "The holdout test set stays untouched until final evaluation. Cross-validation and probability calibration are both fitted only on the training split."
    )

    if not st.button("Train Classification Models", type="primary", width="stretch"):
        st.info("Choose target settings and feature columns, then train the classifiers.")
        return

    if not target_column and not score_columns:
        st.error("Training cannot start because no score columns were selected for the Pass/Fail target.")
        return

    if not feature_columns:
        st.error("Training cannot start because no feature columns were selected.")
        return

    try:
        with st.spinner("Training calibrated classification models and evaluation diagnostics..."):
            run = train_models(
                data=data,
                feature_columns=feature_columns,
                target_column=target_column,
                score_columns=score_columns,
                pass_threshold=pass_threshold,
                test_size=test_size_percent / 100,
                random_state=int(random_state),
                cv_folds=int(cv_folds),
            )
    except Exception as error:
        st.error(f"Classification training failed: {error}")
        return

    set_classification_run(run)
    st.success("Decision Tree, Logistic Regression, and Random Forest models trained successfully.")

    metric_columns = st.columns(5)
    metric_columns[0].metric("Training Rows", f"{run.train_rows:,}")
    metric_columns[1].metric("Testing Rows", f"{run.test_rows:,}")
    metric_columns[2].metric("Features Used", f"{len(run.feature_columns):,}")
    metric_columns[3].metric("CV Folds", f"{run.cv_folds}")
    metric_columns[4].metric("Target", PASS_FAIL_TARGET)

    st.subheader("Model Performance Comparison")
    st.dataframe(build_model_metrics_table(run.results), width="stretch", hide_index=True)
    st.caption(run.performance_summary)
    render_model_explanation_card(
        model_name="Classification Engine",
        explanation="This module compares three supervised classifiers on a held-out test set after leakage-safe preprocessing. Cross-validation and calibration diagnostics are used when the dataset is large enough.",
        confidence_text="Trust is higher when holdout performance, cross-validation stability, and calibration quality all support the same model behavior.",
        governance_text="The dashboard removes target-defining leakage columns and keeps test records unseen until final evaluation.",
    )
    render_preprocessing_audit_card()

    details_tab, evaluation_tab, class_tab = st.tabs(
        ["Engine Details", "Evaluation Visuals", "Class Distribution"]
    )

    with details_tab:
        st.write(f"Target source: `{run.target_source}`")
        st.write(f"Preprocessing: {run.preprocessing_summary}")
        st.write("Numeric features:")
        st.write(", ".join(run.numeric_feature_columns) if run.numeric_feature_columns else "None")
        st.write("Categorical features:")
        st.write(", ".join(run.categorical_feature_columns) if run.categorical_feature_columns else "None")
        st.write("Final training feature set:")
        st.write(", ".join(run.feature_columns))
        st.write("Leakage-protected columns removed:")
        st.write(", ".join(run.removed_leakage_columns) if run.removed_leakage_columns else "None")
        st.write("Probability calibration:")
        if run.calibration_folds is not None and run.calibration_folds >= 2:
            st.write(f"Enabled with {run.calibration_folds} training folds.")
        else:
            st.write("Skipped because the active training split was too small.")
        st.caption(
            "Holdout metrics are calculated on unseen test data. Cross-validation metrics summarize training-fold stability and probability calibration diagnostics show how reliable the predicted Pass probabilities are."
        )

    with evaluation_tab:
        for result in run.results:
            render_model_evaluation(result)

    with class_tab:
        st.dataframe(run.class_distribution, width="stretch", hide_index=True)
        feature_overview = build_feature_importance_overview(run)
        if not feature_overview.empty:
            st.subheader("Cross-Model Feature Importance Overview")
            st.dataframe(feature_overview, width="stretch", hide_index=True)
