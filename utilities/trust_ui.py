"""Shared trust, explainability, and governance UI helpers.

These helpers keep institutional-trust messaging consistent across the
dashboard by centralizing dataset source, preprocessing audit, model
explainability, and prediction disclaimer rendering.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from utilities.dataset_manager import get_dataset_name, get_dataset_origin, get_preprocessing_summary, get_schema_mapping


def get_dataset_source_summary() -> dict[str, object]:
    """Return a compact summary of the active dataset's provenance."""
    mapping = get_schema_mapping() or {}
    preprocessing_summary = get_preprocessing_summary()
    dataset_origin = get_dataset_origin()
    dataset_name = get_dataset_name()

    governance_flags = {
        "schema_mapped": bool(mapping),
        "preprocessed": preprocessing_summary is not None,
        "source_recorded": bool(dataset_origin),
    }
    return {
        "dataset_name": dataset_name,
        "dataset_origin": dataset_origin,
        "schema_mapping": mapping,
        "preprocessing_summary": preprocessing_summary,
        "governance_flags": governance_flags,
    }


def render_dataset_source_banner() -> None:
    """Display the active dataset source and handling state."""
    summary = get_dataset_source_summary()
    origin_label = str(summary["dataset_origin"]).replace("_", " ").title()
    governance_flags = summary["governance_flags"]
    chips = [
        "Schema mapped" if governance_flags["schema_mapped"] else "Schema not mapped",
        "Preprocessed" if governance_flags["preprocessed"] else "Preprocessing pending",
        "Session-scoped source" if governance_flags["source_recorded"] else "Source unavailable",
    ]
    st.markdown(
        f"""
        <div class="module-intro">
            <strong>Dataset source: {escape(str(summary["dataset_name"]))}</strong>
            <span>{escape(origin_label)} | {' | '.join(chips)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_preprocessing_audit_rows(summary) -> list[tuple[str, str]]:
    """Convert a preprocessing summary object into a readable audit list."""
    if summary is None:
        return [("Audit status", "No preprocessing audit has been generated for this session yet.")]

    rows = [
        ("Rows processed", f"{summary.processed_rows:,}"),
        ("Duplicates removed", f"{summary.duplicate_rows_removed:,}"),
        ("Missing values before", f"{summary.missing_values_before:,}"),
        ("Missing values after", f"{summary.missing_values_after:,}"),
        ("Identifier columns excluded", ", ".join(summary.identifier_columns) if getattr(summary, "identifier_columns", None) else "None"),
        (
            "High-cardinality columns excluded",
            ", ".join(summary.high_cardinality_columns) if getattr(summary, "high_cardinality_columns", None) else "None",
        ),
        ("Feature matrix format", getattr(summary, "feature_matrix_format", "Unknown")),
    ]
    return rows


def render_preprocessing_audit_card() -> None:
    """Render a compact preprocessing audit summary for governance review."""
    summary = get_preprocessing_summary()
    rows = build_preprocessing_audit_rows(summary)
    body = "".join(
        f"<div class='trust-audit-row'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
        for label, value in rows
    )
    st.markdown(
        f"""
        <section class="dashboard-panel">
            <h3>Preprocessing Audit Summary</h3>
            <p>The dashboard records how raw student records were cleaned before analytics or modeling outputs were generated.</p>
            <div class="trust-audit-grid">{body}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_data_governance_indicators() -> None:
    """Render high-level governance indicators for academic reviewers."""
    summary = get_dataset_source_summary()
    governance_flags = summary["governance_flags"]
    mapping = summary["schema_mapping"]
    metrics = st.columns(4)
    metrics[0].metric("Dataset Source", "Tracked" if governance_flags["source_recorded"] else "Unknown")
    metrics[1].metric("Schema Mapping", "Ready" if governance_flags["schema_mapped"] else "Pending")
    metrics[2].metric("Preprocessing Audit", "Ready" if governance_flags["preprocessed"] else "Pending")
    metrics[3].metric(
        "Mapped Roles",
        str(
            sum(
                1
                for value in mapping.values()
                if value not in (None, [], {}, "Not mapped")
            )
        )
        if mapping
        else "0",
    )


def render_model_explanation_card(model_name: str, explanation: str, confidence_text: str, governance_text: str) -> None:
    """Render a reusable model explanation card."""
    st.markdown(
        f"""
        <section class="dashboard-panel">
            <h3>{escape(model_name)} Explanation</h3>
            <p>{escape(explanation)}</p>
            <div class="trust-audit-grid">
                <div class="trust-audit-row"><span>Confidence interpretation</span><strong>{escape(confidence_text)}</strong></div>
                <div class="trust-audit-row"><span>Governance note</span><strong>{escape(governance_text)}</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def explain_prediction_confidence(pass_probability: float, fail_probability: float) -> str:
    """Translate raw class probabilities into plain-language confidence guidance."""
    probability_gap = abs(pass_probability - fail_probability)
    winning_probability = max(pass_probability, fail_probability)
    if probability_gap < 0.1:
        return "Low separation between outcomes; this case should be reviewed carefully."
    if winning_probability < 0.7:
        return "Moderate confidence; this is better treated as an early-warning signal than a final judgment."
    return "Higher confidence; the model shows a clearer separation between Pass and Fail for this profile."


def render_prediction_disclaimers() -> None:
    """Render academic-safe disclaimers for prediction usage."""
    st.markdown(
        """
        <section class="dashboard-panel">
            <h3>Prediction Use Guidance</h3>
            <p>Predictions are decision-support signals, not final academic decisions. They should be reviewed alongside attendance records, assessment context, lecturer judgment, and student support policies.</p>
            <div class="trust-audit-grid">
                <div class="trust-audit-row"><span>Appropriate use</span><strong>Early intervention, risk triage, academic support planning</strong></div>
                <div class="trust-audit-row"><span>Not appropriate</span><strong>Automatic grading, disciplinary action, or exclusion decisions</strong></div>
                <div class="trust-audit-row"><span>Confidence meaning</span><strong>Confidence reflects model separation on the available data, not certainty about the student</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_feature_importance_overview(run) -> pd.DataFrame:
    """Build a compact cross-model feature-importance summary."""
    if run is None or not getattr(run, "results", None):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for result in run.results:
        feature_table = getattr(result, "feature_importance", pd.DataFrame())
        if feature_table is None or feature_table.empty:
            continue
        top_rows = feature_table.head(5)
        for _, row in top_rows.iterrows():
            rows.append(
                {
                    "Model": result.model_name,
                    "Feature": row["Feature"],
                    "Importance": round(float(row["Importance"]), 4),
                }
            )
    return pd.DataFrame(rows)
