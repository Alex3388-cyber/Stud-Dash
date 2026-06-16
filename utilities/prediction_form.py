"""Professional Streamlit prediction console for student Pass/Fail outcomes."""

from __future__ import annotations

from html import escape
from math import ceil

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.student_prediction import (
    FEATURE_LABELS,
    FORM_FEATURES,
    FAIL_LABEL,
    PASS_LABEL,
    StudentPrediction,
)
from services.database_service import save_prediction, record_audit_event
from services.prediction_service import infer_feature_mapping, prepare_prediction_model, run_prediction
from utilities.dataset_manager import get_schema_mapping
from utilities.trust_ui import (
    explain_prediction_confidence,
    render_dataset_source_banner,
    render_model_explanation_card,
    render_prediction_disclaimers,
)
from utilities.validation import display_validation_errors, validate_prediction_inputs
from visualizations.plotly_theme import apply_premium_chart_theme


def get_training_dataset(active_data: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, str]:
    """Return the dataset used to train the form's Logistic Regression model."""
    return active_data, dataset_name


def get_risk_class(risk_level: str) -> str:
    """Return the CSS class used to color-code the risk indicator."""
    normalized_risk = risk_level.lower()
    if "low" in normalized_risk:
        return "risk-low"
    if "moderate" in normalized_risk:
        return "risk-moderate"
    return "risk-high"


def normalize_field_name(name: str) -> str:
    """Normalize a mapped dataset column name for display logic."""
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def get_numeric_column_default(data: pd.DataFrame, column: str, fallback: float) -> float:
    """Use the median uploaded value as a sensible default for prediction sliders."""
    if column not in data.columns:
        return fallback
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        return fallback
    return float(values.median())


def build_prediction_field_config(
    feature: str,
    mapped_column: str,
    training_data: pd.DataFrame,
    feature_mode: str = "direct",
) -> dict[str, float | str]:
    """Create dynamic slider labels/ranges from the uploaded dataset mapping."""
    normalized_column = normalize_field_name(mapped_column)

    if feature == "absences":
        values = pd.to_numeric(training_data[mapped_column], errors="coerce").dropna()
        maximum = float(max(20, ceil(values.max()))) if not values.empty else 20.0
        if feature_mode == "attendance_ratio_proxy":
            maximum = 100.0
        elif feature_mode == "attendance_percent_proxy":
            maximum = 100.0

        title = "Absences"
        caption = "This mapped field is used as an attendance-related academic risk signal."
        if feature_mode == "attendance_ratio_proxy":
            title = "Attendance Risk Proxy"
            caption = "This dataset provides attendance as a 0-1 ratio, so the console converts lower attendance into a higher risk signal."
        elif feature_mode == "attendance_percent_proxy":
            title = "Attendance Risk Proxy"
            caption = "This dataset provides attendance as a percentage, so the console converts lower attendance into a higher risk signal."
        elif normalized_column not in {"absences", "absence", "classesmissed"}:
            title = str(mapped_column)

        return {
            "title": title,
            "range": f"0-{maximum:g}",
            "caption": caption,
            "min": 0.0,
            "max": maximum,
            "value": min(get_numeric_column_default(training_data, mapped_column, 4.0), maximum)
            if feature_mode == "absence_count"
            else min(
                max(
                    (1 - get_numeric_column_default(training_data, mapped_column, 0.8)) * 100
                    if feature_mode == "attendance_ratio_proxy"
                    else 100 - get_numeric_column_default(training_data, mapped_column, 80.0),
                    0.0,
                ),
                maximum,
            ),
            "step": 1.0,
            "accent": "input-teal",
        }

    if feature == "study_time":
        values = pd.to_numeric(training_data[mapped_column], errors="coerce").dropna()
        minimum = float(values.min()) if not values.empty else 0.0
        maximum = float(max(minimum + 1, values.max())) if not values.empty else 10.0
        step = 1.0 if values.empty or float(values.dropna().mod(1).sum()) == 0 else 0.5
        return {
            "title": "Study Time" if normalized_column == "studytime" else str(mapped_column),
            "range": f"{minimum:g}-{maximum:g}",
            "caption": "This mapped field represents study effort or study-time intensity.",
            "min": minimum,
            "max": maximum,
            "value": min(max(get_numeric_column_default(training_data, mapped_column, minimum), minimum), maximum),
            "step": step,
            "accent": "input-violet",
        }

    if feature == "failures":
        if feature_mode == "default_zero":
            return {
                "title": "Past Failures",
                "range": "0-4",
                "caption": "No dedicated failure-history column was found, so the console uses a neutral default baseline.",
                "min": 0.0,
                "max": 4.0,
                "value": 0.0,
                "step": 1.0,
                "accent": "input-rose",
            }
        values = pd.to_numeric(training_data[mapped_column], errors="coerce").dropna()
        maximum = float(max(4, ceil(values.max()))) if not values.empty else 4.0
        return {
            "title": "Past Failures" if normalized_column == "failures" else str(mapped_column),
            "range": f"0-{maximum:g}",
            "caption": "This mapped field is used as an academic history or failure-risk signal.",
            "min": 0.0,
            "max": maximum,
            "value": min(get_numeric_column_default(training_data, mapped_column, 0.0), maximum),
            "step": 1.0,
            "accent": "input-rose",
        }

    if feature in {"previous_grade_1", "previous_grade_2"}:
        title_lookup = {
            "g1": "First Period Grade (G1)",
            "g2": "Second Period Grade (G2)",
            "g3": "Final Grade (G3)",
        }
        title = title_lookup.get(normalized_column, str(mapped_column))
        values = pd.to_numeric(training_data[mapped_column], errors="coerce").dropna()
        maximum = float(max(20, ceil(values.max()))) if not values.empty else 20.0
        return {
            "title": title,
            "range": f"0-{maximum:g}",
            "caption": "This mapped score field is used as a prior academic performance signal.",
            "min": 0.0,
            "max": maximum,
            "value": min(max(get_numeric_column_default(training_data, mapped_column, min(10.0, maximum)), 0.0), maximum),
            "step": 1.0,
            "accent": "input-cyan" if feature == "previous_grade_1" else "input-amber",
        }
    return {
        "title": str(mapped_column),
        "range": "0-100",
        "caption": "Mapped academic input.",
        "min": 0.0,
        "max": 100.0,
        "value": 0.0,
        "step": 1.0,
        "accent": "input-cyan",
    }


def build_prediction_probability_gauge(prediction: StudentPrediction) -> go.Figure:
    """Build a dark themed gauge showing the model's Pass probability."""
    pass_percent = prediction.pass_probability * 100
    gauge_color = "#48e69b" if prediction.predicted_label == PASS_LABEL else "#ff776d"

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pass_percent,
            number={
                "suffix": "%",
                "font": {"color": "#edf7ff", "size": 42},
                "valueformat": ".1f",
            },
            title={
                "text": "Pass Probability",
                "font": {"color": "#a8bad0", "size": 15},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#a8bad0",
                    "tickfont": {"color": "#a8bad0", "size": 11},
                },
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(8,18,35,0.35)",
                "bordercolor": "rgba(182,226,255,0.22)",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,119,109,0.22)"},
                    {"range": [50, 75], "color": "rgba(255,209,102,0.2)"},
                    {"range": [75, 100], "color": "rgba(72,230,155,0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#65c7ff", "width": 4},
                    "thickness": 0.78,
                    "value": 50,
                },
            },
        )
    )

    apply_premium_chart_theme(figure, height=330)
    figure.update_layout(
        margin={"l": 18, "r": 18, "t": 52, "b": 18},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def render_prediction_result(prediction: StudentPrediction) -> None:
    """Display the Pass/Fail output, probability, risk level, and recommendation."""
    status_class = "prediction-pass" if prediction.predicted_label == PASS_LABEL else "prediction-fail"
    risk_class = get_risk_class(prediction.risk_level)
    probability_percent = prediction.pass_probability * 100
    fail_percent = prediction.fail_probability * 100
    confidence_percent = prediction.confidence_score * 100

    st.markdown(
        """
        <div class="section-divider">
            <span></span>
            <strong>Prediction Output</strong>
            <span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_left, result_right = st.columns([0.95, 1.05])
    with result_left:
        st.markdown(
            f"""
            <div class="prediction-outcome-card {status_class}">
                <div class="prediction-card-kicker">AI Model Decision</div>
                <strong>{escape(prediction.predicted_label)}</strong>
                <p>The Logistic Regression model compares the mapped academic risk signals and prior performance fields against learned Pass/Fail patterns.</p>
                <div class="probability-pair">
                    <span>Pass <b>{probability_percent:.1f}%</b></span>
                    <span>Fail <b>{fail_percent:.1f}%</b></span>
                </div>
                <div class="prediction-confidence-card">
                    <span>Confidence</span>
                    <strong>{confidence_percent:.1f}%</strong>
                </div>
                <div class="probability-track">
                    <span style="width: {probability_percent:.1f}%"></span>
                </div>
                <div class="risk-indicator {risk_class}">
                    <i></i>
                    <span>{escape(prediction.risk_level)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_right:
        st.plotly_chart(build_prediction_probability_gauge(prediction), width="stretch")

    st.markdown(
        f"""
        <div class="recommendation-card {risk_class}">
            <span>Recommended Academic Action</span>
            <p>{escape(prediction.recommendation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence_explanation = explain_prediction_confidence(
        pass_probability=prediction.pass_probability,
        fail_probability=prediction.fail_probability,
    )
    render_model_explanation_card(
        model_name="Logistic Regression",
        explanation="The prediction model estimates Pass or Fail by comparing the mapped study, attendance, prior-grade, and academic-history inputs with patterns learned from the active dataset.",
        confidence_text=confidence_explanation,
        governance_text="Predictions should support early academic intervention, not replace lecturer judgment or institutional review processes.",
    )
    render_prediction_disclaimers()


def render_student_prediction_form(active_data: pd.DataFrame, dataset_name: str) -> None:
    """Render the prediction form and train/use the Logistic Regression model."""
    render_dataset_source_banner()
    safe_dataset_name = escape(dataset_name)
    st.markdown(
        f"""
        <div class="prediction-console-hero">
            <div>
                <span class="prediction-console-kicker">AI Prediction Console</span>
                <h2>Live Student Performance Intelligence</h2>
                <p>Run a live Logistic Regression prediction using the active schema-mapped academic inputs to estimate Pass or Fail outcomes.</p>
            </div>
            <div class="console-live-chip">
                <span class="status-dot"></span>
                <strong>Model Ready</strong>
                <small>{safe_dataset_name}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    training_data, training_source = get_training_dataset(active_data, dataset_name)
    schema_mapping = get_schema_mapping()
    feature_mapping = infer_feature_mapping(training_data, schema_mapping=schema_mapping)
    missing_features = [feature for feature in ["study_time", "absences", "previous_grade_1", "previous_grade_2"] if feature not in feature_mapping]

    if missing_features:
        readable_missing = ", ".join(FEATURE_LABELS[feature] for feature in missing_features)
        mapped_columns = ", ".join(feature_mapping.values()) if feature_mapping else "None"
        st.warning(
            "This uploaded dataset does not include all fields needed for the live prediction console."
        )
        st.caption(f"Missing prediction fields: {readable_missing}")
        st.caption(f"Detected compatible columns: {mapped_columns}")
        return

    try:
        model_bundle = prepare_prediction_model(training_data, schema_mapping=schema_mapping)
    except Exception as error:
        st.error(f"Unable to prepare the Logistic Regression model: {error}")
        return

    field_configs = {
        feature: build_prediction_field_config(
            feature,
            model_bundle.feature_mapping[feature],
            training_data,
            feature_mode=model_bundle.feature_modes.get(feature, "direct"),
        )
        for feature in FORM_FEATURES
    }

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(
            """
            <section class="prediction-input-panel">
                <div class="panel-title-row">
                    <div>
                        <span>Input Matrix</span>
                        <strong>Academic Risk Signals</strong>
                    </div>
                    <div class="panel-chip">5 Features</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("student_prediction_console"):
            first_row = st.columns(2)
            with first_row[0]:
                config = field_configs["study_time"]
                st.markdown(
                    f"""
                    <div class="prediction-input-card {escape(str(config["accent"]))}">
                        <div class="input-card-top"><span>{escape(str(config["title"]))}</span><strong>{escape(str(config["range"]))}</strong></div>
                        <p>{escape(str(config["caption"]))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                study_time = st.slider(str(config["title"]), min_value=float(config["min"]), max_value=float(config["max"]), value=float(config["value"]), step=float(config["step"]))
            with first_row[1]:
                config = field_configs["absences"]
                st.markdown(
                    f"""
                    <div class="prediction-input-card {escape(str(config["accent"]))}">
                        <div class="input-card-top"><span>{escape(str(config["title"]))}</span><strong>{escape(str(config["range"]))}</strong></div>
                        <p>{escape(str(config["caption"]))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                absences = st.slider(str(config["title"]), min_value=float(config["min"]), max_value=float(config["max"]), value=float(config["value"]), step=float(config["step"]))

            second_row = st.columns(2)
            with second_row[0]:
                config = field_configs["failures"]
                st.markdown(
                    f"""
                    <div class="prediction-input-card {escape(str(config["accent"]))}">
                        <div class="input-card-top"><span>{escape(str(config["title"]))}</span><strong>{escape(str(config["range"]))}</strong></div>
                        <p>{escape(str(config["caption"]))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                failures = st.slider(str(config["title"]), min_value=float(config["min"]), max_value=float(config["max"]), value=float(config["value"]), step=float(config["step"]))
            with second_row[1]:
                config = field_configs["previous_grade_1"]
                st.markdown(
                    f"""
                    <div class="prediction-input-card {escape(str(config["accent"]))}">
                        <div class="input-card-top"><span>{escape(str(config["title"]))}</span><strong>{escape(str(config["range"]))}</strong></div>
                        <p>{escape(str(config["caption"]))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                previous_grade_1 = st.slider(str(config["title"]), min_value=float(config["min"]), max_value=float(config["max"]), value=float(config["value"]), step=float(config["step"]))

            third_row = st.columns(1)
            with third_row[0]:
                config = field_configs["previous_grade_2"]
                st.markdown(
                    f"""
                    <div class="prediction-input-card {escape(str(config["accent"]))}">
                        <div class="input-card-top"><span>{escape(str(config["title"]))}</span><strong>{escape(str(config["range"]))}</strong></div>
                        <p>{escape(str(config["caption"]))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                previous_grade_2 = st.slider(str(config["title"]), min_value=float(config["min"]), max_value=float(config["max"]), value=float(config["value"]), step=float(config["step"]))

            submitted = st.form_submit_button("Run Live AI Prediction", type="primary", width="stretch")

        st.markdown("</section>", unsafe_allow_html=True)

    with right:
        st.markdown(
            f"""
            <section class="prediction-model-panel">
                <div class="panel-title-row">
                    <div>
                        <span>Model Telemetry</span>
                        <strong>Logistic Regression</strong>
                    </div>
                    <div class="model-orb">AI</div>
                </div>
                <div class="model-stat-grid">
                    <div><span>Training Source</span><strong>{escape(training_source)}</strong></div>
                    <div><span>Training Rows</span><strong>{model_bundle.training_rows:,}</strong></div>
                    <div><span>Target Source</span><strong>{escape(model_bundle.target_source)}</strong></div>
                    <div><span>Output</span><strong>Pass / Fail</strong></div>
                </div>
                <div class="risk-scale-card">
                    <span>Risk Thresholds</span>
                    <p><b>Low:</b> 75%+ pass chance</p>
                    <p><b>Moderate:</b> 50-74.9% pass chance</p>
                    <p><b>High:</b> below 50% pass chance</p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        render_model_explanation_card(
            model_name="Prediction Confidence",
            explanation="The displayed confidence is based on the model's estimated Pass and Fail probabilities. A narrow gap between those probabilities means the case is less clear-cut.",
            confidence_text="Confidence is strongest when one outcome probability clearly exceeds the other.",
            governance_text="Probability output is a statistical estimate from the uploaded data, not a guarantee about an individual student's result.",
        )

    if not submitted:
        st.markdown(
            """
            <div class="prediction-awaiting-card">
                <span class="status-dot"></span>
                <div>
                    <strong>Console awaiting input</strong>
                    <p>Adjust the live academic sliders, then run the prediction to see Pass/Fail outcome, confidence, risk level, and recommendation.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    prediction_inputs = {
        "study_time": study_time,
        "absences": absences,
        "failures": failures,
        "previous_grade_1": previous_grade_1,
        "previous_grade_2": previous_grade_2,
    }
    validation_constraints = {
        feature: {
            "min": float(config["min"]),
            "max": float(config["max"]),
            "title": str(config["title"]),
        }
        for feature, config in field_configs.items()
    }
    validation_errors = validate_prediction_inputs(prediction_inputs, field_constraints=validation_constraints)
    if validation_errors:
        display_validation_errors(validation_errors)
        return

    try:
        prediction = run_prediction(model_bundle=model_bundle, **prediction_inputs)
    except Exception as error:
        st.error("Prediction could not be generated. Check the input values and model training data.")
        st.caption(f"Technical detail: {error}")
        return

    render_prediction_result(prediction)

    try:
        prediction_id = save_prediction(
            dataset_name=dataset_name,
            model_name="Logistic Regression",
            inputs=prediction_inputs,
            prediction=prediction,
        )
        st.session_state["last_prediction_id"] = prediction_id
        st.caption(f"Prediction saved to SQLite with ID `{prediction_id}`.")
        try:
            record_audit_event(
                "prediction",
                entity_name=dataset_name,
                detail=f"Logistic Regression → {prediction.predicted_label} ({prediction.confidence_score * 100:.1f}% confidence)",
                rows_affected=1,
            )
        except Exception:
            pass
    except Exception as error:
        st.warning("Prediction completed, but it could not be saved to SQLite.")
        st.caption(f"Technical detail: {error}")
