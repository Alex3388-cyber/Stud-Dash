"""Home page rendering for the premium student analytics dashboard."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from models.ai_insights import AiInsight, build_ai_insights
from models.classification import FAIL_LABEL, PASS_LABEL
from services.analytics_service import (
    KpiCard,
    build_home_kpis,
    calculate_row_average_scores,
    format_percent,
    get_total_students,
    scale_percent_value,
)
from services.dataset_service import ensure_active_dataset_pipeline, get_dataset_origin_label, get_schema_value, numeric_series
from ui.shell import extract_numeric_display_value, lucide_icon, render_panel
from utilities.database_ui import render_saved_records_dashboard
from utilities.schema_mapping import normalize_lookup
from utilities.trust_ui import (
    render_data_governance_indicators,
    render_dataset_source_banner,
    render_model_explanation_card,
    render_preprocessing_audit_card,
)
from utilities.dataset_upload import CSV_EXCEL_FILE_TYPES, process_uploaded_dataset_file
from visualizations.plotly_theme import PREMIUM_COLORWAY, PREMIUM_DIVERGING_SCALE, apply_premium_chart_theme


RISK_CATEGORY_COLORS = {
    "High Performers": PREMIUM_COLORWAY[5],
    "Average Progress": PREMIUM_COLORWAY[2],
    "Average Performers": PREMIUM_COLORWAY[2],
    "At-Risk Students": PREMIUM_COLORWAY[4],
}

HOME_UPLOAD_WIDGET_KEYS = [
    "home_dataset_upload",
    "home_upload_preview_rows",
]


def render_home_kpi_cards(cards: list[KpiCard]) -> str:
    """Render the KPI card grid as HTML."""
    card_markup = []
    for card in cards:
        count_value, prefix, suffix = extract_numeric_display_value(card.value)
        count_attributes = (
            f'data-value="{count_value}" data-prefix="{escape(prefix)}" data-suffix="{escape(suffix)}"'
            if count_value is not None
            else f'data-static="{escape(card.value)}"'
        )
        card_markup.append(
            (
                f'<article class="home-kpi-card kpi-{escape(card.accent)}">'
                '<div class="kpi-glow-ring"></div>'
                f'<span class="kpi-icon">{escape(card.icon)}</span>'
                f"<small>{escape(card.title)}</small>"
                f'<strong class="kpi-count" {count_attributes}>{escape(card.value)}</strong>'
                f"<em>{escape(card.note)}</em>"
                '<div class="kpi-progress-line"><span></span></div>'
                '<div class="kpi-shine"></div>'
                "</article>"
            )
        )
    return "".join(card_markup)


def render_animated_kpi_component(cards: list[KpiCard]) -> None:
    """Render KPI cards with CSS effects and JavaScript count-up animation."""
    cards_markup = render_home_kpi_cards(cards)
    st.html(
        f"""
        <style>
            :root {{
                --card-bg:   #ffffff;
                --card-bdr:  #e5e7eb;
                --icon-bg:   #eff6ff;
                --text:      #111827;
                --muted:     #6b7280;
                --muted-2:   #9ca3af;
                --blue:      #3b82f6;
                --green:     #22c55e;
                --red:       #ef4444;
                --amber:     #f59e0b;
                --purple:    #a855f7;
                --teal:      #14b8a6;
                --radius:    10px;
            }}

            * {{ box-sizing: border-box; }}

            body {{
                background: transparent;
                font-family: "Inter", "Segoe UI", system-ui, sans-serif;
                line-height: 1.55;
                margin: 0;
                overflow: hidden;
                -webkit-font-smoothing: antialiased;
            }}

            .home-kpi-grid {{
                display: grid;
                gap: 0.75rem;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                padding: 2px;
            }}

            .home-kpi-card {{
                background: var(--card-bg);
                border: 1px solid var(--card-bdr);
                border-left: 3px solid var(--blue);
                border-radius: var(--radius);
                box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 4px 12px rgba(0,0,0,.06);
                display: flex;
                flex-direction: column;
                min-height: 130px;
                overflow: hidden;
                padding: 1rem;
                position: relative;
                animation: kpi-rise 400ms cubic-bezier(0.16, 1, 0.3, 1) forwards;
                opacity: 0;
                transform: translateY(8px);
                transition: box-shadow 150ms ease, transform 150ms ease;
            }}

            .home-kpi-card:nth-child(2) {{ animation-delay: 60ms; }}
            .home-kpi-card:nth-child(3) {{ animation-delay: 120ms; }}
            .home-kpi-card:nth-child(4) {{ animation-delay: 180ms; }}
            .home-kpi-card:nth-child(5) {{ animation-delay: 240ms; }}
            .home-kpi-card:nth-child(6) {{ animation-delay: 300ms; }}
            .home-kpi-card:nth-child(7) {{ animation-delay: 360ms; }}

            .home-kpi-card:hover {{
                box-shadow: 0 4px 12px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.08);
                transform: translateY(-2px);
            }}

            .home-kpi-card.kpi-green  {{ border-left-color: var(--green); }}
            .home-kpi-card.kpi-rose,
            .home-kpi-card.kpi-danger {{ border-left-color: var(--red); }}
            .home-kpi-card.kpi-violet {{ border-left-color: var(--purple); }}
            .home-kpi-card.kpi-amber  {{ border-left-color: var(--amber); }}
            .home-kpi-card.kpi-cyan   {{ border-left-color: var(--teal); }}
            .home-kpi-card.kpi-teal   {{ border-left-color: var(--teal); }}

            .kpi-glow-ring,
            .kpi-shine {{ display: none !important; }}

            .kpi-icon {{
                align-items: center;
                background: var(--icon-bg);
                border-radius: 6px;
                color: var(--blue);
                display: flex;
                font-size: 0.72rem;
                font-weight: 700;
                height: 30px;
                justify-content: center;
                margin-bottom: 0.5rem;
                width: 30px;
            }}

            .kpi-green .kpi-icon, .kpi-teal .kpi-icon   {{ background: #f0fdf4; color: var(--green); }}
            .kpi-rose .kpi-icon,  .kpi-danger .kpi-icon {{ background: #fef2f2; color: var(--red); }}
            .kpi-violet .kpi-icon {{ background: #faf5ff; color: var(--purple); }}
            .kpi-amber .kpi-icon  {{ background: #fffbeb; color: var(--amber); }}
            .kpi-cyan .kpi-icon   {{ background: #f0fdfa; color: var(--teal); }}

            .home-kpi-card small {{
                color: var(--muted);
                display: block;
                font-size: 0.67rem;
                font-weight: 600;
                letter-spacing: .04em;
                line-height: 1.25;
                min-height: 1.45rem;
                text-transform: uppercase;
            }}

            .home-kpi-card strong {{
                color: var(--text);
                display: block;
                font-size: 1.35rem;
                font-weight: 700;
                line-height: 1.12;
                margin: 0.15rem 0 0.18rem;
                overflow-wrap: anywhere;
            }}

            .home-kpi-card em {{
                color: var(--muted-2);
                display: block;
                font-size: 0.74rem;
                font-style: normal;
                line-height: 1.38;
                min-height: 1.8rem;
            }}

            .kpi-progress-line {{
                background: #f3f4f6;
                border-radius: 999px;
                height: 3px;
                margin-top: auto;
                overflow: hidden;
            }}

            .kpi-progress-line span {{
                animation: kpi-progress 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
                background: var(--blue);
                border-radius: inherit;
                display: block;
                height: 100%;
                transform: scaleX(0);
                transform-origin: left;
            }}

            .kpi-green .kpi-progress-line span, .kpi-teal .kpi-progress-line span  {{ background: var(--green); }}
            .kpi-rose .kpi-progress-line span, .kpi-danger .kpi-progress-line span {{ background: var(--red); }}
            .kpi-violet .kpi-progress-line span {{ background: var(--purple); }}
            .kpi-amber .kpi-progress-line span  {{ background: var(--amber); }}
            .kpi-cyan .kpi-progress-line span   {{ background: var(--teal); }}

            @keyframes kpi-rise {{
                from {{ opacity: 0; transform: translateY(8px); }}
                to   {{ opacity: 1; transform: translateY(0); }}
            }}

            @keyframes kpi-progress {{
                to {{ transform: scaleX(1); }}
            }}

            @media (max-width: 1100px) {{
                .home-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            }}

            @media (max-width: 760px) {{
                body {{ overflow: auto; }}
                .home-kpi-grid {{ grid-template-columns: 1fr; }}
                .home-kpi-card {{ min-height: 116px; }}
            }}

            @media (prefers-reduced-motion: reduce) {{
                *, *::before, *::after {{
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }}
            }}
        </style>
        <section class="home-kpi-grid">{cards_markup}</section>
        <script>
            const easeOutCubic = (value) => 1 - Math.pow(1 - value, 3);
            const formatValue = (value, suffix) => {{
                const decimals = suffix === "%" ? 1 : 0;
                return value.toLocaleString(undefined, {{
                    minimumFractionDigits: decimals,
                    maximumFractionDigits: decimals
                }});
            }};

            document.querySelectorAll(".kpi-count").forEach((element) => {{
                const staticValue = element.dataset.static;
                if (staticValue) {{
                    element.textContent = staticValue;
                    return;
                }}

                const target = Number(element.dataset.value || "0");
                const prefix = element.dataset.prefix || "";
                const suffix = element.dataset.suffix || "";
                const duration = 1300;
                const start = performance.now();

                const tick = (now) => {{
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = easeOutCubic(progress);
                    const current = target * eased;
                    element.textContent = `${{prefix}}${{formatValue(current, suffix)}}${{suffix}}`;
                    if (progress < 1) {{
                        requestAnimationFrame(tick);
                    }} else {{
                        element.textContent = `${{prefix}}${{formatValue(target, suffix)}}${{suffix}}`;
                    }}
                }};

                requestAnimationFrame(tick);
            }});
        </script>
        """,
        unsafe_allow_javascript=True,
        width="stretch",
    )


def render_home_kpi_fallback(cards: list[KpiCard]) -> None:
    """Render static KPI cards if the components iframe is unavailable."""
    st.markdown(
        f'<div class="home-kpi-grid">{render_home_kpi_cards(cards)}</div>',
        unsafe_allow_html=True,
    )


def format_hero_metric(value: object, fallback: str = "--%") -> str:
    """Format dynamic hero metrics safely."""
    if value is None:
        return fallback
    if isinstance(value, float):
        return format_percent(value)
    return str(value)


def build_hero_bar_markup(row_average_scores: pd.Series) -> str:
    """Build hero mini-bars from real score distribution values."""
    clean_scores = row_average_scores.dropna() if isinstance(row_average_scores, pd.Series) else pd.Series(dtype="float64")
    if clean_scores.empty:
        heights = [8, 8, 8, 8, 8, 8]
    else:
        bins = pd.cut(clean_scores.clip(lower=0, upper=100), bins=[0, 40, 50, 60, 70, 85, 100], include_lowest=True)
        counts = bins.value_counts(sort=False)
        max_count = max(int(counts.max()), 1)
        heights = [max(10, round((int(count) / max_count) * 92)) for count in counts]
    return "".join(f'<span style="height: {height}%"></span>' for height in heights)


def build_home_dataset_quick_summary(data: pd.DataFrame) -> dict[str, int | float]:
    """Build a lightweight dataset summary for the Home upload status card."""
    row_count = len(data)
    column_count = len(data.columns)
    total_missing = int(data.isna().sum().sum())
    numeric_columns = len(data.select_dtypes(include="number").columns)
    categorical_columns = len(data.select_dtypes(exclude="number").columns)
    completeness = (
        100.0
        if row_count == 0 or column_count == 0
        else 100 - ((total_missing / max(row_count * column_count, 1)) * 100)
    )
    return {
        "row_count": row_count,
        "column_count": column_count,
        "total_missing": total_missing,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "completeness": completeness,
    }


def render_home_upload_card(snapshot: dict[str, object]) -> dict[str, object]:
    """Render the Home page upload experience and return upload/pipeline state."""
    active_data = snapshot.get("data")
    dataset_name = str(snapshot.get("dataset_name", "Uploaded dataset"))
    source_label = str(snapshot.get("source_label", "Upload required"))
    has_active_data = isinstance(active_data, pd.DataFrame) and not active_data.empty

    quick_summary = build_home_dataset_quick_summary(active_data) if has_active_data else {}
    row_count = int(quick_summary.get("row_count", 0))
    column_count = int(quick_summary.get("column_count", 0))
    total_missing = int(quick_summary.get("total_missing", 0))
    completeness = float(quick_summary.get("completeness", 100.0))

    upload_result: dict[str, object] | None = None
    pipeline_ran = False
    pipeline_report = None
    request_rerun = False

    left, right = st.columns([1.12, 0.88])
    with left:
        st.markdown(
            f"""
            <section class="hero-upload-card">
                <div class="hero-upload-header">
                    <div class="hero-upload-icon">{lucide_icon("cloud-upload", "hero-upload-icon-svg")}</div>
                    <div>
                        <span class="hero-upload-kicker">Live Dataset Intake</span>
                        <h3>Upload a student dataset.</h3>
                        <p>CSV or Excel files refresh analytics, KPIs, and model outputs for this session.</p>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drop a student dataset here",
            type=CSV_EXCEL_FILE_TYPES,
            help="Drag and drop a CSV, XLSX, or XLS file to refresh the dashboard.",
            key="home_dataset_upload",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            upload_result = process_uploaded_dataset_file(
                uploaded_file,
                allowed_file_types=CSV_EXCEL_FILE_TYPES,
                widget_keys=HOME_UPLOAD_WIDGET_KEYS,
            )
            if upload_result.get("errors"):
                for message in upload_result["errors"]:
                    st.error(str(message))
            elif upload_result.get("error"):
                st.error(str(upload_result["error"]))
                if upload_result.get("error_detail"):
                    st.caption(f"Technical detail: {upload_result['error_detail']}")
            else:
                uploaded_name = str(upload_result.get("dataset_name", uploaded_file.name))
                uploaded_data = upload_result.get("data")
                st.success(f"Dataset `{uploaded_name}` loaded successfully.")
                if upload_result.get("saved_dataset_id") is not None:
                    st.caption(f"Persisted to SQLite with ID `{upload_result['saved_dataset_id']}`.")
                if upload_result.get("save_warning"):
                    st.warning(str(upload_result["save_warning"]))
                    if upload_result.get("save_error_detail"):
                        st.caption(f"Technical detail: {upload_result['save_error_detail']}")

                if isinstance(uploaded_data, pd.DataFrame) and not uploaded_data.empty:
                    pipeline_ran, pipeline_report = ensure_active_dataset_pipeline(show_spinner=True)
                    request_rerun = True

        st.markdown("</section>", unsafe_allow_html=True)

    with right:
        dataset_badge = "Dataset ready" if has_active_data else "Awaiting upload"
        status_class = "is-ready" if has_active_data else "is-empty"
        st.markdown(
            f"""
            <section class="hero-dataset-status-card {status_class}">
                <div class="hero-dataset-status-top">
                    <span class="hero-dataset-status-pill"><i></i>{escape(dataset_badge)}</span>
                    <small>{escape(source_label)}</small>
                </div>
                <strong>{escape(dataset_name)}</strong>
                <div class="hero-dataset-metrics">
                    <div><span>Rows</span><strong>{row_count:,}</strong></div>
                    <div><span>Columns</span><strong>{column_count:,}</strong></div>
                    <div><span>Missing</span><strong>{total_missing:,}</strong></div>
                    <div><span>Completeness</span><strong>{completeness:.1f}%</strong></div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if has_active_data:
            numeric_columns = int(quick_summary.get("numeric_columns", 0))
            categorical_columns = int(quick_summary.get("categorical_columns", 0))
            score_columns = snapshot.get("score_columns", []) or []
            preview_rows = min(max(row_count, 1), 3)
            st.markdown(
                f"""
                <div class="hero-dataset-summary-grid">
                    <div><span>Numeric</span><strong>{numeric_columns}</strong></div>
                    <div><span>Categorical</span><strong>{categorical_columns}</strong></div>
                    <div><span>Score Fields</span><strong>{len(score_columns)}</strong></div>
                    <div><span>Schema</span><strong>{"Mapped" if score_columns else "Scanning"}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Quick dataset preview")
            st.dataframe(active_data.head(preview_rows), width="stretch", height=150)
        else:
            st.markdown(
                """
                <div class="hero-dataset-empty-copy">
                    <p>Upload a dataset to activate live analytics and model outputs.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</section>", unsafe_allow_html=True)

    if upload_result and pipeline_ran and pipeline_report:
        st.success(
            "Dashboard analytics refreshed. "
            f"Preprocessing: {pipeline_report['preprocessing']} | "
            f"Classification: {pipeline_report['classification']} | "
            f"Clustering: {pipeline_report['clustering']}"
        )

    if upload_result and not pipeline_ran and pipeline_report:
        st.info(
            "The active dataset was already synchronized with the analytics pipeline."
        )

    return {
        "upload_result": upload_result,
        "pipeline_ran": pipeline_ran,
        "pipeline_report": pipeline_report,
        "request_rerun": request_rerun,
    }


def render_home_hero(snapshot: dict[str, object]) -> None:
    """Render the premium dashboard hero for the Home page."""
    pass_rate = format_hero_metric(snapshot["pass_rate"])
    fail_rate = format_hero_metric(snapshot["fail_rate"])
    average_score = format_hero_metric(snapshot["average_score"])
    attendance_metric_title = escape(str(snapshot.get("attendance_metric_title", "Attendance Metric")))
    attendance_metric_value = format_hero_metric(
        snapshot.get("average_absences") if snapshot.get("average_absences") is not None else snapshot.get("average_attendance")
    )
    prediction_accuracy = format_hero_metric(snapshot["prediction_accuracy"])
    at_risk_count = "--" if snapshot["at_risk_count"] is None else f"{snapshot['at_risk_count']:,}"
    dataset_name = escape(str(snapshot.get("dataset_name", "Uploaded dataset")))
    hero_bars = build_hero_bar_markup(snapshot.get("row_average_scores", pd.Series(dtype="float64")))

    st.markdown(
        f"""
        <section class="home-hero">
            <div class="hero-gradient-layer"></div>
            <div class="home-hero-copy">
                <div class="hero-badge"><span class="status-dot"></span> Student PP Intelligence Suite</div>
                <h1>Student performance analytics for faster academic decisions.</h1>
                <p>
                    Active dataset: {dataset_name}. Monitor performance trends, upload records, and review risk signals from one focused dashboard.
                </p>
                <div class="hero-chip-row">
                    <span>Performance Metrics</span>
                    <span>Attendance Signals</span>
                    <span>Risk Patterns</span>
                </div>
            </div>
            <div class="home-hero-visual" aria-hidden="true">
                <div class="visual-header">
                    <span class="status-dot"></span>
                    <strong>Live Prediction Console</strong>
                </div>
                <div class="visual-score">
                    <span>Average Score</span>
                    <strong>{average_score}</strong>
                </div>
                <div class="visual-bars">
                    {hero_bars}
                </div>
                <div class="visual-insight-grid">
                    <span>Pass {pass_rate}</span>
                    <span>Fail {fail_rate}</span>
                    <span>Risk {at_risk_count}</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        render_animated_kpi_component(snapshot["cards"])
    except Exception:
        render_home_kpi_fallback(snapshot["cards"])

    insight_columns = st.columns(3)
    insight_columns[0].caption("Prediction Insight")
    insight_columns[0].markdown(f"**{prediction_accuracy} accuracy**")
    insight_columns[1].caption("Performance Summary")
    insight_columns[1].markdown(f"**{average_score} score | {attendance_metric_value} {attendance_metric_title.lower()}**")
    insight_columns[2].caption("Student Risk")
    insight_columns[2].markdown(f"**{at_risk_count} at-risk students**")


def render_ai_insight_card(insight: AiInsight) -> str:
    """Render one AI insight card as HTML."""
    return (
        f'<article class="ai-insight-card insight-{escape(insight.severity)}">'
        '<div class="ai-card-top">'
        f'<span class="ai-insight-icon">{escape(insight.icon)}</span>'
        f'<small>{escape(insight.label)}</small>'
        "</div>"
        f"<strong>{escape(insight.title)}</strong>"
        f"<p>{escape(insight.message)}</p>"
        f'<div class="ai-recommendation"><span>Recommendation</span>{escape(insight.recommendation)}</div>'
        "</article>"
    )


def render_ai_insights_panel(snapshot: dict[str, object]) -> None:
    """Render dynamic AI academic insight cards."""
    insights = build_ai_insights(
        snapshot,
        get_total_students=get_total_students,
        format_percent=format_percent,
        normalize_lookup=normalize_lookup,
    )
    insights_markup = "".join(render_ai_insight_card(insight) for insight in insights)
    st.markdown(
        f"""
        <div class="section-divider">
            <span></span>
            <strong>AI Academic Insights</strong>
            <span></span>
        </div>
        <section class="ai-insights-panel">
            <div class="ai-panel-header">
                <div>
                    <span class="ai-panel-kicker">AI Analytics Engine</span>
                    <h3>Dynamic academic intelligence from the active dataset</h3>
                </div>
                <div class="ai-panel-status"><span class="status-dot"></span> Live scan</div>
            </div>
            <div class="ai-insight-grid">{insights_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_ai_insights_preview(snapshot: dict[str, object], limit: int = 3) -> None:
    """Render a compact preview of the highest-priority AI insights on Home."""
    insights = build_ai_insights(
        snapshot,
        get_total_students=get_total_students,
        format_percent=format_percent,
        normalize_lookup=normalize_lookup,
    )[:limit]
    if not insights:
        return

    preview_markup = "".join(render_ai_insight_card(insight) for insight in insights)
    st.markdown(
        f"""
        <div class="section-divider">
            <span></span>
            <strong>AI Insight Preview</strong>
            <span></span>
        </div>
        <section class="ai-insights-panel">
            <div class="ai-panel-header">
                <div>
                    <span class="ai-panel-kicker">Priority Signals</span>
                    <h3>Top academic alerts from the active dataset</h3>
                </div>
                <div class="ai-panel-status"><span class="status-dot"></span> Preview</div>
            </div>
            <div class="ai-insight-grid">{preview_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_chart_empty_state(message: str) -> None:
    """Render a styled empty state when a chart cannot be generated."""
    st.markdown(
        f"""
        <div class="chart-empty-state">
            <span>Chart unavailable</span>
            <p>{escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pass_fail_donut(snapshot: dict[str, object]) -> None:
    """Render Pass vs Fail donut chart."""
    pass_count = snapshot.get("pass_count")
    fail_count = snapshot.get("fail_count")
    if pass_count is None or fail_count is None:
        render_chart_empty_state("Pass/Fail values need a target column or score fields.")
        return

    chart_data = pd.DataFrame(
        {
            "Outcome": [PASS_LABEL, FAIL_LABEL],
            "Students": [int(pass_count), int(fail_count)],
        }
    )
    figure = px.pie(
        chart_data,
        names="Outcome",
        values="Students",
        hole=0.62,
        title="Pass vs Fail Distribution",
        color="Outcome",
        color_discrete_map={PASS_LABEL: PREMIUM_COLORWAY[5], FAIL_LABEL: PREMIUM_COLORWAY[4]},
    )
    figure.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.02, 0.04],
        hovertemplate="<b>%{label}</b><br>Students: %{value}<br>Share: %{percent}<extra></extra>",
    )
    st.plotly_chart(apply_premium_chart_theme(figure, height=390), width="stretch")


def render_average_subject_scores(snapshot: dict[str, object]) -> None:
    """Render a bar chart of average scores per detected subject field."""
    data = snapshot["data"]
    score_columns = snapshot["score_columns"]
    if not score_columns:
        render_chart_empty_state("Detected score columns are required for the average subject score chart.")
        return

    score_values = data.loc[:, score_columns].apply(pd.to_numeric, errors="coerce")
    average_scores = score_values.mean(axis=0).dropna()
    if average_scores.empty:
        render_chart_empty_state("Average subject scores could not be calculated from the current dataset.")
        return

    max_score = score_values.stack().max()
    scale_factor = 100
    scale_label = "Average Score (%)"
    if max_score == max_score and float(max_score) <= 1.5:
        average_scores = average_scores * 100
    elif max_score == max_score and float(max_score) <= 20:
        average_scores = average_scores * 5
    else:
        scale_factor = 1
        scale_label = "Average Score"

    chart_data = (
        average_scores.rename("Average Score")
        .rename_axis("Subject")
        .reset_index()
        .sort_values("Average Score", ascending=False)
    )
    figure = px.bar(
        chart_data,
        x="Subject",
        y="Average Score",
        color="Average Score",
        color_continuous_scale=PREMIUM_DIVERGING_SCALE,
        title="Average Subject Score",
    )
    figure.update_traces(
        hovertemplate="<b>%{x}</b><br>Average: %{y:.1f}" + ("%" if scale_factor == 100 else "") + "<extra></extra>"
    )
    figure.update_layout(coloraxis_colorbar_title=scale_label)
    st.plotly_chart(apply_premium_chart_theme(figure, height=390), width="stretch")


def render_attendance_score_scatter(snapshot: dict[str, object]) -> None:
    """Render engagement or absence vs score scatter chart."""
    row_average_scores = snapshot["row_average_scores"]
    attendance_column = snapshot["attendance_column"]
    data = snapshot["data"]
    if not attendance_column or row_average_scores.empty or row_average_scores.dropna().empty:
        render_chart_empty_state("Absence/attendance and score columns are required for this scatter plot.")
        return

    attendance_values = numeric_series(data, attendance_column)
    is_absence_column = normalize_lookup(attendance_column) in {"absences", "absence", "classesmissed"}
    x_label = "Absences" if is_absence_column else "Attendance"
    x_values = attendance_values if is_absence_column else attendance_values.map(
        lambda value: scale_percent_value(value, attendance_values)
    )
    scatter_data = pd.DataFrame(
        {
            x_label: x_values,
            "Average Score": row_average_scores,
            "Risk Category": snapshot["risk_categories"],
        }
    ).dropna(subset=[x_label, "Average Score"])

    if scatter_data.empty:
        render_chart_empty_state("Absence/attendance and score values could not be converted to numbers.")
        return

    figure = px.scatter(
        scatter_data,
        x=x_label,
        y="Average Score",
        color="Risk Category",
        size="Average Score",
        size_max=18,
        title=f"{x_label} vs Score",
        color_discrete_map=RISK_CATEGORY_COLORS,
    )
    figure.update_traces(
        customdata=scatter_data[["Risk Category"]],
        hovertemplate=f"<b>{x_label}</b>: %{{x}}<br>Average Score: %{{y:.1f}}%<br>Risk: %{{customdata[0]}}<extra></extra>",
    )
    st.plotly_chart(apply_premium_chart_theme(figure, height=390), width="stretch")


def render_score_distribution(snapshot: dict[str, object]) -> None:
    """Render the score distribution histogram."""
    row_average_scores = snapshot["row_average_scores"]
    if row_average_scores.empty or row_average_scores.dropna().empty:
        render_chart_empty_state("Score columns are required for the score distribution chart.")
        return

    chart_data = pd.DataFrame({"Average Score": row_average_scores}).dropna()
    figure = px.histogram(
        chart_data,
        x="Average Score",
        nbins=18,
        title="Score Distribution",
        marginal="rug",
        color_discrete_sequence=[PREMIUM_COLORWAY[1]],
    )
    figure.update_traces(hovertemplate="<b>Score Bin</b><br>Average Score: %{x}<br>Students: %{y}<extra></extra>")
    st.plotly_chart(apply_premium_chart_theme(figure, height=390), width="stretch")


def render_risk_category_bar(snapshot: dict[str, object]) -> None:
    """Render the risk category bar chart."""
    risk_categories = snapshot["risk_categories"]
    if risk_categories.empty:
        render_chart_empty_state("Risk categories need score, attendance, or Pass/Fail data.")
        return

    chart_data = risk_categories.dropna().value_counts().rename_axis("Risk Category").reset_index(name="Students")
    if chart_data.empty:
        render_chart_empty_state("Risk categories could not be calculated from the current data.")
        return

    figure = px.bar(
        chart_data,
        x="Risk Category",
        y="Students",
        color="Risk Category",
        title="Risk Category Breakdown",
        color_discrete_map=RISK_CATEGORY_COLORS,
    )
    figure.update_traces(hovertemplate="<b>%{x}</b><br>Students: %{y}<extra></extra>")
    st.plotly_chart(apply_premium_chart_theme(figure, height=390), width="stretch")


def render_correlation_heatmap(snapshot: dict[str, object]) -> None:
    """Render a correlation heatmap from the uploaded student dataset."""
    data = snapshot["data"]
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if len(numeric_columns) < 2:
        render_chart_empty_state("At least two numeric student fields are required for the correlation heatmap.")
        return

    preferred_order = [
        column
        for column in (get_schema_value("clustering_feature_columns", []) or []) + (get_schema_value("performance_columns", []) or [])
        if column in data.columns and column in numeric_columns
    ]
    preferred_order = list(dict.fromkeys(preferred_order))
    remaining_columns = [column for column in numeric_columns if column not in preferred_order]
    selected_columns = (preferred_order + remaining_columns)[: min(6, len(numeric_columns))]
    correlation_matrix = data.loc[:, selected_columns].corr()
    if correlation_matrix.empty:
        render_chart_empty_state("Correlation values could not be calculated from the current numeric fields.")
        return

    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=PREMIUM_DIVERGING_SCALE,
        title="Correlation Heatmap",
    )
    figure.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>")
    st.plotly_chart(apply_premium_chart_theme(figure, height=430), width="stretch")


def render_home_analytics_charts(snapshot: dict[str, object]) -> None:
    """Render interactive analytics charts on the Home page."""
    data = snapshot["data"]
    score_columns = snapshot.get("score_columns", []) or []
    has_target = snapshot.get("target") is not None and not snapshot["target"].dropna().empty
    attendance_column = snapshot.get("attendance_column")
    numeric_columns = data.select_dtypes(include="number").columns.tolist() if isinstance(data, pd.DataFrame) else []
    can_show_any_chart = bool(
        has_target
        or score_columns
        or (attendance_column and score_columns)
        or len(numeric_columns) >= 2
    )

    if data.empty:
        return

    if not can_show_any_chart:
        st.info("Charts will appear here once the uploaded dataset includes mapped score, attendance, or numeric academic fields.")
        return

    st.markdown(
        """
        <div class="section-divider">
            <span></span>
            <strong>Interactive Student Analytics</strong>
            <span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    first_left, first_right = st.columns([0.9, 1.1])
    with first_left:
        render_pass_fail_donut(snapshot)
    with first_right:
        render_average_subject_scores(snapshot)

    second_left, second_right = st.columns(2)
    with second_left:
        render_attendance_score_scatter(snapshot)
    with second_right:
        render_score_distribution(snapshot)

    third_left, third_right = st.columns([0.92, 1.08])
    with third_left:
        render_risk_category_bar(snapshot)
    with third_right:
        render_correlation_heatmap(snapshot)


def render_home() -> None:
    """Render the home dashboard overview."""
    ensure_active_dataset_pipeline(show_spinner=False)
    snapshot = build_home_kpis()
    has_data = not snapshot["data"].empty

    if not has_data:
        st.markdown(
            """
            <div class="status-banner">
                <span class="status-dot"></span>
                <strong>Upload required</strong>
                <span>Upload a student dataset to populate KPIs, charts, insights, preprocessing, prediction, clustering, and reports with real values.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        active_source = get_dataset_origin_label()
        st.markdown(
            f"""
            <div class="status-banner">
                <span class="status-dot"></span>
                <strong>Active dataset synced</strong>
                <span>{escape(str(snapshot['dataset_name']))} | {len(snapshot['data']):,} record(s) | {active_source}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_home_hero(snapshot)
    upload_state = render_home_upload_card(snapshot)
    if bool(upload_state.get("request_rerun")):
        st.rerun()
    overview_tab, analytics_tab, governance_tab = st.tabs(["Overview", "Analytics", "Trust & Records"])

    with overview_tab:
        summary_left, summary_right = st.columns([1.15, 0.85])
        with summary_left:
            render_ai_insights_preview(snapshot)
        with summary_right:
            render_dataset_source_banner()
            render_data_governance_indicators()

        if not has_data:
            st.markdown(
                """
                <div class="section-divider">
                    <span></span>
                    <strong>Getting Started</strong>
                    <span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            with left:
                render_panel(
                    "Next Step",
                    "Start by loading a student dataset into the dashboard.",
                    [
                        "Upload CSV or Excel from the Home page",
                        "Map academic fields after upload",
                        "Review KPIs and insights",
                    ],
                )
            with right:
                render_panel(
                    "Recommended Flow",
                    "Move through the dashboard in a clean academic workflow.",
                    [
                        "Home for upload and overview",
                        "Data Exploration for patterns",
                        "Prediction and Clustering for modeling outputs",
                    ],
                )

    with analytics_tab:
        render_home_analytics_charts(snapshot)

    with governance_tab:
        render_model_explanation_card(
            model_name="Dashboard Trust Summary",
            explanation="Dashboard analytics are generated from the active session dataset, schema mapping, preprocessing audit, and the latest trained models. Each module works from the same centralized dataset state.",
            confidence_text="KPI and chart trust is highest when the dataset source is mapped, preprocessing has completed, and model outputs are available from the same session.",
            governance_text="Reports are designed to use human-readable datasets, while transformed feature matrices are kept separate for modeling workflows.",
        )
        render_preprocessing_audit_card()
        with st.expander("Saved Records", expanded=False):
            render_saved_records_dashboard()
