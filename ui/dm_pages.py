"""Data Mining, Data Warehousing, and Decision Support page renderers.

All 10 new phases are implemented here as standalone render functions.
Each page gracefully handles the empty-dataset state.
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.dataset_service import get_active_kpi_dataset
from ui.shell import render_page_header, render_panel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_data() -> tuple[pd.DataFrame, str]:
    """Return the active dashboard dataset (empty DataFrame if none loaded)."""
    data, name, _src = get_active_kpi_dataset()
    return data, name or "Dataset"


def _no_data_banner() -> bool:
    """Show a prompt and return True when no dataset is loaded."""
    data, _ = _get_data()
    if isinstance(data, pd.DataFrame) and not data.empty:
        return False
    st.info("Upload a student dataset from the **Home** page to populate this view.")
    return True


def _metric_row(items: list[tuple[str, str, str]]) -> None:
    """Render a row of st.metric cards from (label, value, delta) tuples."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta if delta else None)


# ---------------------------------------------------------------------------
# Phase 1 — Data Warehouse Architecture
# ---------------------------------------------------------------------------

def render_data_warehouse() -> None:
    """Visualise the four-layer warehouse architecture with record counts."""
    render_page_header(
        "Data Warehouse",
        "Raw, Cleaned, Transformed, and Warehouse data layers with record counts and quality indicators.",
        "DW Architecture",
    )

    from services.database_service import get_dataset_uploads, get_record_counts

    counts = get_record_counts()

    # ── Layer pipeline diagram ──────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0 24px">
          <div style="flex:1;min-width:140px;background:rgba(15,52,96,.6);border:1px solid #1e4d7e;border-radius:8px;padding:16px;text-align:center">
            <div style="color:#65c7ff;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase">Raw Layer</div>
            <div style="font-size:28px;font-weight:800;margin:6px 0">📥</div>
            <div style="font-size:12px;color:#a8bad0">Original uploads as-is</div>
          </div>
          <div style="color:#36e6c2;font-size:24px">→</div>
          <div style="flex:1;min-width:140px;background:rgba(15,52,96,.6);border:1px solid #1e4d7e;border-radius:8px;padding:16px;text-align:center">
            <div style="color:#36e6c2;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase">Cleaned Layer</div>
            <div style="font-size:28px;font-weight:800;margin:6px 0">🧹</div>
            <div style="font-size:12px;color:#a8bad0">Dedup, null fill, type coercion</div>
          </div>
          <div style="color:#36e6c2;font-size:24px">→</div>
          <div style="flex:1;min-width:140px;background:rgba(15,52,96,.6);border:1px solid #1e4d7e;border-radius:8px;padding:16px;text-align:center">
            <div style="color:#9b7cff;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase">Transformed Layer</div>
            <div style="font-size:28px;font-weight:800;margin:6px 0">⚙️</div>
            <div style="font-size:12px;color:#a8bad0">Scaled, encoded, feature matrix</div>
          </div>
          <div style="color:#36e6c2;font-size:24px">→</div>
          <div style="flex:1;min-width:140px;background:rgba(15,52,96,.6);border:1px solid #1e4d7e;border-radius:8px;padding:16px;text-align:center">
            <div style="color:#ffd166;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase">Warehouse Layer</div>
            <div style="font-size:28px;font-weight:800;margin:6px 0">🏛️</div>
            <div style="font-size:12px;color:#a8bad0">KPIs, models, audit tables</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Storage metrics ─────────────────────────────────────────────────────
    _metric_row([
        ("Datasets Uploaded", str(counts.get("datasets", 0)), None),
        ("Stored Rows", f"{counts.get('rows', 0):,}", None),
        ("Predictions Saved", str(counts.get("predictions", 0)), None),
    ])

    st.divider()

    # ── Active dataset layer view ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Upload Registry")
        uploads = get_dataset_uploads()
        if uploads.empty:
            st.info("No datasets ingested yet.")
        else:
            uploads_display = uploads[["dataset_id", "file_name", "row_count", "column_count", "created_at"]].copy()
            uploads_display.columns = ["ID", "File", "Rows", "Columns", "Uploaded"]
            st.dataframe(uploads_display, use_container_width=True, hide_index=True)

    with col2:
        data, dataset_name = _get_data()
        st.subheader("Active Data Layer Status")
        if isinstance(data, pd.DataFrame) and not data.empty:
            from utilities.dataset_manager import get_cleaned_dataset, get_feature_matrix
            raw_rows = len(data)
            cleaned = get_cleaned_dataset()
            cleaned_rows = len(cleaned) if cleaned is not None else raw_rows
            feature_mat = get_feature_matrix()
            transformed_cols = feature_mat.shape[1] if feature_mat is not None else 0

            layer_df = pd.DataFrame([
                {"Layer": "Raw", "Records": raw_rows, "Columns": len(data.columns), "Status": "✅ Available"},
                {"Layer": "Cleaned", "Records": cleaned_rows, "Columns": len(data.columns), "Status": "✅ Available" if cleaned is not None else "⏳ Pending"},
                {"Layer": "Transformed", "Records": cleaned_rows if cleaned is not None else raw_rows,
                 "Columns": transformed_cols, "Status": "✅ Available" if feature_mat is not None else "⏳ Pending"},
                {"Layer": "Warehouse", "Records": counts.get("rows", 0), "Columns": "—", "Status": "✅ Persisted" if counts.get("rows", 0) > 0 else "⏳ Pending"},
            ])
            st.dataframe(layer_df, use_container_width=True, hide_index=True)
        else:
            st.info("Upload a dataset to see active layer status.")


# ---------------------------------------------------------------------------
# Phase 2 — ETL Monitoring Dashboard
# ---------------------------------------------------------------------------

def render_etl_monitor() -> None:
    """Show ETL pipeline execution history, timings, and record counts."""
    render_page_header(
        "ETL Monitor",
        "Track extraction, transformation, and loading stages with timings, row counts, and error details.",
        "Pipeline Monitor",
    )

    from services.database_service import get_etl_jobs, get_dataset_uploads

    jobs = get_etl_jobs(limit=100)
    uploads = get_dataset_uploads()

    if jobs.empty:
        st.info(
            "No ETL jobs recorded yet. "
            "Upload a dataset to run the extraction → cleaning → transformation → load pipeline "
            "and populate this monitor."
        )
        _render_pipeline_status_key()
        return

    # ── Summary metrics ─────────────────────────────────────────────────────
    total_jobs = len(jobs)
    completed = (jobs["status"] == "completed").sum()
    failed = (jobs["status"] == "failed").sum()
    avg_ms = int(jobs["duration_ms"].dropna().mean()) if not jobs["duration_ms"].dropna().empty else 0

    _metric_row([
        ("Total Pipeline Runs", str(total_jobs), None),
        ("Completed", str(completed), None),
        ("Failed", str(failed), f"-{failed}" if failed else None),
        ("Avg Duration", f"{avg_ms} ms", None),
    ])

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Pipeline Job History")
        display = jobs[["job_id", "dataset_name", "stage", "rows_in", "rows_out",
                        "rows_rejected", "duration_ms", "status", "started_at"]].copy()
        display.columns = ["ID", "Dataset", "Stage", "Rows In", "Rows Out",
                          "Rejected", "Duration (ms)", "Status", "Started"]
        st.dataframe(display, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Stage Distribution")
        if "stage" in jobs.columns and not jobs["stage"].dropna().empty:
            stage_counts = jobs["stage"].value_counts().reset_index()
            stage_counts.columns = ["Stage", "Count"]
            fig = px.pie(stage_counts, names="Stage", values="Count",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # ── Records processed over time ─────────────────────────────────────────
    if "rows_out" in jobs.columns and "started_at" in jobs.columns:
        st.subheader("Records Processed Over Time")
        jobs_sorted = jobs.sort_values("started_at")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=jobs_sorted["started_at"], y=jobs_sorted["rows_in"],
                              name="Rows In", marker_color="#65c7ff"))
        fig2.add_trace(go.Bar(x=jobs_sorted["started_at"], y=jobs_sorted["rows_out"],
                              name="Rows Out", marker_color="#36e6c2"))
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", barmode="group",
                          xaxis_title="Run Time", yaxis_title="Record Count")
        st.plotly_chart(fig2, use_container_width=True)


def _render_pipeline_status_key() -> None:
    st.markdown(
        """
        <div style="background:rgba(15,52,96,.4);border:1px solid #1e4d7e;border-radius:8px;padding:16px;margin-top:16px">
          <strong>Pipeline Stages Tracked</strong>
          <ul style="margin:8px 0 0;color:#a8bad0;font-size:13px">
            <li><b>Extract</b> — Read raw file bytes and parse CSV/Excel</li>
            <li><b>Validate</b> — Schema check, type detection, quality scoring</li>
            <li><b>Transform</b> — Dedup, null fill, scaling, encoding</li>
            <li><b>Load</b> — Persist rows to SQLite warehouse</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Phase 3 — Data Quality Dashboard
# ---------------------------------------------------------------------------

def render_data_quality() -> None:
    """Completeness, duplicates, outliers, and overall quality score."""
    render_page_header(
        "Data Quality",
        "Completeness scores, duplicate rates, outlier detection, and column-level quality indicators.",
        "Quality Assessment",
    )

    if _no_data_banner():
        return

    data, dataset_name = _get_data()

    # ── Quality score computation ────────────────────────────────────────────
    total_cells = data.size
    missing_cells = int(data.isna().sum().sum())
    completeness = max(0.0, 1 - missing_cells / max(total_cells, 1))

    dup_rows = int(data.duplicated().sum())
    dup_rate = dup_rows / max(len(data), 1)
    dup_score = max(0.0, 1 - dup_rate)

    numeric_data = data.select_dtypes(include="number")
    outlier_cols = []
    if not numeric_data.empty:
        try:
            from scipy import stats as scipy_stats
            z = numeric_data.apply(lambda col: (col - col.mean()) / col.std(ddof=0)
                                   if col.std(ddof=0) > 0 else pd.Series(0, index=col.index))
            outlier_flags = (z.abs() > 3)
            outlier_cols = [c for c in outlier_flags.columns if outlier_flags[c].any()]
        except Exception:
            # Manual z-score fallback
            for col in numeric_data.columns:
                series = numeric_data[col].dropna()
                if len(series) > 0 and series.std() > 0:
                    z_vals = (series - series.mean()) / series.std()
                    if (z_vals.abs() > 3).any():
                        outlier_cols.append(col)

    outlier_score = max(0.0, 1 - len(outlier_cols) / max(len(numeric_data.columns), 1))

    # Weighted overall score
    overall = 0.4 * completeness + 0.3 * dup_score + 0.3 * outlier_score
    grade = "A" if overall >= 0.9 else "B" if overall >= 0.75 else "C" if overall >= 0.6 else "D"
    grade_color = "#36e6c2" if overall >= 0.75 else "#ffd166" if overall >= 0.6 else "#ff6b91"

    # ── Score card ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex;gap:16px;align-items:stretch;flex-wrap:wrap;margin-bottom:24px">
          <div style="flex:1;min-width:120px;background:rgba(15,52,96,.6);border:1px solid {grade_color};border-radius:8px;padding:20px;text-align:center">
            <div style="color:{grade_color};font-size:11px;font-weight:700;letter-spacing:1px">QUALITY GRADE</div>
            <div style="font-size:52px;font-weight:900;color:{grade_color}">{grade}</div>
            <div style="font-size:13px;color:#a8bad0">{overall*100:.1f} / 100</div>
          </div>
          <div style="flex:1;min-width:120px;background:rgba(15,52,96,.6);border:1px solid #1e4d7e;border-radius:8px;padding:20px;text-align:center">
            <div style="color:#65c7ff;font-size:11px;font-weight:700;letter-spacing:1px">COMPLETENESS</div>
            <div style="font-size:36px;font-weight:800;color:#65c7ff">{completeness*100:.1f}%</div>
            <div style="font-size:12px;color:#a8bad0">{missing_cells:,} missing cells</div>
          </div>
          <div style="flex:1;min-width:120px;background:rgba(15,52,96,.6);border:1px solid #1e4d7e;border-radius:8px;padding:20px;text-align:center">
            <div style="color:#ffd166;font-size:11px;font-weight:700;letter-spacing:1px">DUPLICATE ROWS</div>
            <div style="font-size:36px;font-weight:800;color:#ffd166">{dup_rows}</div>
            <div style="font-size:12px;color:#a8bad0">{dup_rate*100:.1f}% of total rows</div>
          </div>
          <div style="flex:1;min-width:120px;background:rgba(15,52,96,.6);border:1px solid #1e4d7e;border-radius:8px;padding:20px;text-align:center">
            <div style="color:#ff6b91;font-size:11px;font-weight:700;letter-spacing:1px">OUTLIER COLUMNS</div>
            <div style="font-size:36px;font-weight:800;color:#ff6b91">{len(outlier_cols)}</div>
            <div style="font-size:12px;color:#a8bad0">z-score &gt; 3 detected</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Column-level completeness bar chart ──────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Column Completeness")
        completeness_df = pd.DataFrame({
            "Column": data.columns,
            "Non-Null %": ((1 - data.isna().mean()) * 100).round(1).values,
        }).sort_values("Non-Null %")
        fig = px.bar(
            completeness_df, x="Non-Null %", y="Column", orientation="h",
            color="Non-Null %", color_continuous_scale="RdYlGn", range_color=[0, 100],
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                         height=max(300, len(data.columns) * 24))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Outlier Columns")
        if outlier_cols:
            st.warning(f"**{len(outlier_cols)} column(s) contain statistical outliers (z > 3):**")
            for col_name in outlier_cols:
                series = numeric_data[col_name].dropna()
                if len(series) > 0 and series.std() > 0:
                    z_series = (series - series.mean()) / series.std()
                    n_outliers = int((z_series.abs() > 3).sum())
                    st.markdown(f"- **{col_name}** — {n_outliers} outlier row(s)")
        else:
            st.success("No statistical outliers detected (z-score > 3) in numeric columns.")

        # ── Data type table ──────────────────────────────────────────────────
        st.subheader("Column Summary")
        quality_df = pd.DataFrame({
            "Column": data.columns,
            "Type": data.dtypes.astype(str).values,
            "Missing": data.isna().sum().values,
            "Unique": data.nunique(dropna=True).values,
            "Missing %": (data.isna().mean().values * 100).round(1),
        })
        st.dataframe(quality_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Phase 4 — Executive KPI Dashboard
# ---------------------------------------------------------------------------

def render_kpi_dashboard() -> None:
    """Materialised KPI cards: pass rate, fail rate, avg score, attendance, segments."""
    render_page_header(
        "KPI Dashboard",
        "Executive view of pass rate, failure rate, attendance, high performers, at-risk students, and departmental breakdown.",
        "Executive KPIs",
    )

    if _no_data_banner():
        return

    from services.analytics_service import (
        build_home_kpis,
        calculate_average_attendance,
        calculate_average_score,
        calculate_pass_fail_kpis,
        calculate_row_average_scores,
        get_pass_fail_target,
        get_total_students,
    )
    from services.dataset_service import get_score_columns, get_attendance_column
    from utilities.schema_mapping import build_auto_schema_mapping

    data, dataset_name = _get_data()
    schema = build_auto_schema_mapping(data)
    score_cols = [c for c in schema.get("score_columns", []) if c in data.columns]
    att_col = schema.get("attendance_column")

    total = get_total_students(data)
    avg_score = calculate_average_score(data, score_cols)
    avg_att = calculate_average_attendance(data, att_col if att_col in data.columns else None)
    target, _col, _src = get_pass_fail_target(data, score_cols)
    pf = calculate_pass_fail_kpis(target)
    row_scores = calculate_row_average_scores(data, score_cols)

    at_risk_count = int((row_scores < 60).sum()) if not row_scores.empty else 0
    high_perf_count = int((row_scores >= 80).sum()) if not row_scores.empty else 0

    # ── Primary KPI row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total:,}")
    c2.metric("Pass Rate", f"{pf.get('pass_rate', 0):.1f}%" if pf.get("pass_rate") is not None else "N/A")
    c3.metric("Fail Rate", f"{pf.get('fail_rate', 0):.1f}%" if pf.get("fail_rate") is not None else "N/A")
    c4.metric("Average Score", f"{avg_score:.1f}%" if avg_score is not None else "N/A")

    # ── Secondary KPI row ────────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("At-Risk Students", str(at_risk_count), f"{at_risk_count/max(total,1)*100:.0f}%")
    c6.metric("High Performers", str(high_perf_count), f"{high_perf_count/max(total,1)*100:.0f}%")
    c7.metric("Attendance", f"{avg_att:.1f}%" if avg_att is not None else "N/A")
    c8.metric("Score Columns", str(len(score_cols)))

    st.divider()

    # ── Score distribution ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        if score_cols and not row_scores.empty:
            st.subheader("Score Distribution")
            fig = px.histogram(
                row_scores, nbins=30, labels={"value": "Average Score", "count": "Students"},
                color_discrete_sequence=["#36e6c2"],
            )
            fig.add_vline(x=60, line_dash="dash", line_color="#ff6b91", annotation_text="Pass Threshold (60%)")
            fig.add_vline(x=80, line_dash="dash", line_color="#ffd166", annotation_text="High Performer (80%)")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                             xaxis_title="Score", yaxis_title="Students")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ── Pass/Fail pie ────────────────────────────────────────────────────
        if pf.get("pass_count") is not None and pf.get("fail_count") is not None:
            st.subheader("Pass / Fail Split")
            fig2 = px.pie(
                values=[pf["pass_count"], pf["fail_count"]],
                names=["Pass", "Fail"],
                color_discrete_sequence=["#36e6c2", "#ff6b91"],
            )
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Departmental breakdown ───────────────────────────────────────────────
    st.subheader("Performance by Group")
    categorical_cols = [c for c in data.columns
                        if data[c].dtype == "object" and 1 < data[c].nunique() <= 12
                        and c not in (score_cols + ([att_col] if att_col else []))]

    if categorical_cols and not row_scores.empty:
        selected_group = st.selectbox("Group by", categorical_cols, key="kpi_group_by")
        group_df = data[[selected_group]].copy()
        group_df["avg_score"] = row_scores
        group_avg = group_df.groupby(selected_group)["avg_score"].agg(["mean", "count"]).reset_index()
        group_avg.columns = [selected_group, "Avg Score", "Students"]
        group_avg = group_avg.sort_values("Avg Score", ascending=False)

        fig3 = px.bar(group_avg, x=selected_group, y="Avg Score",
                     color="Avg Score", color_continuous_scale="RdYlGn",
                     range_color=[0, 100], text="Students",
                     labels={"Avg Score": "Average Score (%)"})
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No suitable categorical grouping columns found for departmental breakdown.")


# ---------------------------------------------------------------------------
# Phase 5 — Explainability
# ---------------------------------------------------------------------------

def render_explainability() -> None:
    """SHAP feature contributions, permutation importance, and counterfactual analysis."""
    render_page_header(
        "Explainability",
        "SHAP values, feature importance, and counterfactual analysis — understand WHY predictions are made.",
        "Explainable AI",
    )

    if _no_data_banner():
        return

    data, dataset_name = _get_data()

    from models.student_prediction import (
        train_logistic_prediction_model,
        predict_student_performance,
        FORM_FEATURES,
        FEATURE_LABELS,
    )
    from models.explainability import explain_prediction, build_counterfactual

    # ── Train model ─────────────────────────────────────────────────────────
    with st.spinner("Training prediction model for explanation…"):
        try:
            bundle = train_logistic_prediction_model(data)
        except Exception as exc:
            st.error(f"Could not train prediction model: {exc}")
            return

    st.success(f"Model trained on {bundle.training_rows:,} records using {bundle.target_source}")

    # ── Input form ───────────────────────────────────────────────────────────
    st.subheader("Enter Student Profile")
    col1, col2, col3, col4, col5 = st.columns(5)
    study_time = col1.number_input("Study Time", 0.0, 20.0, 2.0, step=0.5, key="xai_study")
    absences = col2.number_input("Absences", 0.0, 100.0, 5.0, step=1.0, key="xai_abs")
    failures = col3.number_input("Failures", 0.0, 10.0, 0.0, step=1.0, key="xai_fail")
    g1 = col4.number_input("Grade G1 (0–20)", 0.0, 20.0, 10.0, step=0.5, key="xai_g1")
    g2 = col5.number_input("Grade G2 (0–20)", 0.0, 20.0, 10.0, step=0.5, key="xai_g2")

    if st.button("Explain This Prediction", type="primary"):
        features = pd.DataFrame([{
            "study_time": study_time, "absences": absences, "failures": failures,
            "previous_grade_1": g1, "previous_grade_2": g2,
        }])

        result = predict_student_performance(
            bundle, study_time=study_time, absences=absences, failures=failures,
            previous_grade_1=g1, previous_grade_2=g2,
        )

        # ── Prediction summary ───────────────────────────────────────────────
        pred_col, risk_col, pass_col = st.columns(3)
        pred_col.metric("Prediction", result.predicted_label)
        risk_col.metric("Risk Level", result.risk_level)
        pass_col.metric("Pass Probability", f"{result.pass_probability:.1%}")
        st.info(f"**Recommendation:** {result.recommendation}")

        # ── SHAP / permutation explanation ───────────────────────────────────
        shap_data = explain_prediction(bundle.model, features)
        if shap_data and shap_data.get("top_contributors"):
            st.subheader("Feature Contributions")
            method_label = shap_data.get("method", "permutation").replace("_", " ").title()
            st.caption(f"Method: {method_label}")

            contrib_df = pd.DataFrame(shap_data["top_contributors"])
            contrib_df["direction"] = contrib_df["shap_value"].apply(
                lambda v: "Increases Pass" if v > 0 else "Increases Fail"
            )
            fig = px.bar(
                contrib_df, x="shap_value", y="feature", orientation="h",
                color="shap_value", color_continuous_scale="RdBu_r",
                title="Feature Contributions (positive = pushes toward Pass)",
                labels={"shap_value": "Contribution Score", "feature": "Feature"},
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            # ── Contribution explanation table ───────────────────────────────
            contrib_df["Explanation"] = contrib_df.apply(
                lambda row: (
                    f"{FEATURE_LABELS.get(row['feature'], row['feature'])} is contributing "
                    f"{'positively' if row['shap_value'] > 0 else 'negatively'} to the prediction "
                    f"(score: {row['shap_value']:+.4f})"
                ),
                axis=1,
            )
            st.dataframe(contrib_df[["feature", "shap_value", "Explanation"]], use_container_width=True, hide_index=True)

        # ── Counterfactual analysis ──────────────────────────────────────────
        cf = build_counterfactual(bundle.model, features)
        if cf:
            first = cf[0] if isinstance(cf, list) else cf
            if "message" not in first:
                st.subheader("Counterfactual: What Would Change the Outcome?")
                st.caption("Minimum feature changes required to flip the prediction.")
                st.dataframe(pd.DataFrame(cf) if isinstance(cf, list) else pd.DataFrame([cf]),
                            use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Phase 6 — Advanced Visualizations
# ---------------------------------------------------------------------------

def render_advanced_charts() -> None:
    """Heatmaps, violin plots, pair plots, correlation matrix, and trend analysis."""
    render_page_header(
        "Advanced Charts",
        "Correlation heatmaps, violin plots, pair plots, score trends, and feature distributions.",
        "Visual Analytics",
    )

    if _no_data_banner():
        return

    data, dataset_name = _get_data()
    numeric_cols = data.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for advanced visualizations.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Correlation Heatmap", "Violin Plots", "Score Distributions", "Pair Analysis"])

    with tab1:
        st.subheader("Correlation Matrix Heatmap")
        corr = data[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Between All Numeric Features",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Violin Plots — Score Distributions by Group")
        cat_cols = [c for c in data.columns if data[c].dtype == "object" and 1 < data[c].nunique() <= 10]
        score_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in ("score", "grade", "mark", "result", "g1", "g2", "g3"))]
        if not score_cols:
            score_cols = numeric_cols[:3]

        if cat_cols and score_cols:
            grp = st.selectbox("Group by", cat_cols, key="violin_group")
            metric = st.selectbox("Score metric", score_cols, key="violin_metric")
            fig2 = px.violin(data, x=grp, y=metric, box=True, points="outliers",
                            color=grp, title=f"{metric} Distribution by {grp}")
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Simple violin across all score cols
            melted = data[score_cols].melt(var_name="Feature", value_name="Value")
            fig2 = px.violin(melted, x="Feature", y="Value", box=True, color="Feature")
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Score Histograms")
        selected_cols = st.multiselect("Select columns to plot", numeric_cols, default=numeric_cols[:4], key="hist_cols")
        if selected_cols:
            for idx, col_name in enumerate(selected_cols[:6]):
                fig3 = px.histogram(data, x=col_name, nbins=25,
                                   color_discrete_sequence=["#65c7ff"],
                                   title=f"Distribution of {col_name}")
                fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("Scatter Pair Analysis")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X axis", numeric_cols, key="pair_x")
            y_col = st.selectbox("Y axis", numeric_cols,
                                index=min(1, len(numeric_cols) - 1), key="pair_y")
            color_cols = ["(none)"] + [c for c in data.columns if data[c].dtype == "object" and data[c].nunique() <= 8]
            color_col = st.selectbox("Color by", color_cols, key="pair_color")
            fig4 = px.scatter(
                data, x=x_col, y=y_col,
                color=color_col if color_col != "(none)" else None,
                trendline="ols",
                title=f"{y_col} vs {x_col}",
                opacity=0.65,
            )
            fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Phase 7 — Decision Support
# ---------------------------------------------------------------------------

def render_decision_support() -> None:
    """Auto-generate observations, risk signals, and actionable recommendations."""
    render_page_header(
        "Decision Support",
        "Automated analysis of dataset patterns to generate observations, risk signals, and evidence-based recommendations.",
        "Decision Intelligence",
    )

    if _no_data_banner():
        return

    from services.analytics_service import (
        calculate_average_score,
        calculate_average_attendance,
        calculate_pass_fail_kpis,
        calculate_row_average_scores,
        get_pass_fail_target,
        get_total_students,
    )
    from utilities.schema_mapping import build_auto_schema_mapping

    data, dataset_name = _get_data()
    schema = build_auto_schema_mapping(data)
    score_cols = [c for c in schema.get("score_columns", []) if c in data.columns]
    att_col = schema.get("attendance_column")

    total = get_total_students(data)
    avg_score = calculate_average_score(data, score_cols)
    avg_att = calculate_average_attendance(data, att_col if att_col and att_col in data.columns else None)
    target, _col, _src = get_pass_fail_target(data, score_cols)
    pf = calculate_pass_fail_kpis(target)
    row_scores = calculate_row_average_scores(data, score_cols)

    pass_rate = pf.get("pass_rate") or 0.0
    at_risk_count = int((row_scores < 60).sum()) if not row_scores.empty else 0
    high_perf = int((row_scores >= 80).sum()) if not row_scores.empty else 0

    # ── Build observations ───────────────────────────────────────────────────
    observations: list[dict] = []
    warnings: list[dict] = []
    recommendations: list[dict] = []

    if pass_rate >= 80:
        observations.append({"icon": "✅", "title": "Strong Pass Rate",
                             "text": f"Pass rate stands at {pass_rate:.1f}%, indicating cohort performance is on track."})
    elif pass_rate >= 60:
        observations.append({"icon": "⚠️", "title": "Moderate Pass Rate",
                             "text": f"Pass rate is {pass_rate:.1f}%, which is acceptable but has room for improvement."})
        warnings.append({"icon": "⚠️", "title": "Risk: Below Target",
                        "text": f"Pass rate below 80% — {100 - pass_rate:.1f}% gap to high-performance benchmark."})
    else:
        observations.append({"icon": "🚨", "title": "Low Pass Rate",
                             "text": f"Pass rate is critically low at {pass_rate:.1f}%. Immediate intervention recommended."})
        warnings.append({"icon": "🚨", "title": "Critical Failure Risk",
                        "text": f"Over {100 - pass_rate:.0f}% of students are at risk of failing."})
        recommendations.append({"priority": "HIGH", "action": "Implement targeted tutoring programmes",
                               "rationale": f"Pass rate of {pass_rate:.1f}% indicates systemic academic difficulty."})

    if at_risk_count > 0:
        risk_pct = at_risk_count / max(total, 1) * 100
        if risk_pct > 25:
            warnings.append({"icon": "🚨", "title": "High At-Risk Count",
                            "text": f"{at_risk_count} students ({risk_pct:.1f}%) are scoring below 60%. Urgent support needed."})
            recommendations.append({"priority": "HIGH", "action": "Create at-risk student support groups",
                                   "rationale": f"{at_risk_count} students need immediate academic intervention."})
        else:
            warnings.append({"icon": "⚠️", "title": "Students Below Threshold",
                            "text": f"{at_risk_count} students ({risk_pct:.1f}%) are at risk. Monitor closely."})
            recommendations.append({"priority": "MEDIUM", "action": "Schedule one-to-one progress reviews",
                                   "rationale": f"{at_risk_count} students scoring below passing threshold."})

    if avg_score is not None:
        if avg_score >= 75:
            observations.append({"icon": "✅", "title": "Above-Average Performance",
                                 "text": f"Cohort average score is {avg_score:.1f}%, indicating strong academic outcomes."})
        elif avg_score < 60:
            warnings.append({"icon": "🚨", "title": "Low Average Score",
                            "text": f"Cohort average of {avg_score:.1f}% is below the passing threshold."})
            recommendations.append({"priority": "HIGH", "action": "Review curriculum delivery effectiveness",
                                   "rationale": f"Average score {avg_score:.1f}% suggests curriculum or teaching gaps."})

    if avg_att is not None and avg_att < 75:
        warnings.append({"icon": "⚠️", "title": "Low Attendance",
                        "text": f"Average attendance of {avg_att:.1f}% is below the recommended 75% threshold."})
        recommendations.append({"priority": "HIGH", "action": "Improve attendance monitoring and follow-up",
                               "rationale": f"Low attendance ({avg_att:.1f}%) correlates with poor performance outcomes."})

    if high_perf > 0:
        observations.append({"icon": "⭐", "title": "High Performers Identified",
                             "text": f"{high_perf} students ({high_perf/max(total,1)*100:.1f}%) are scoring above 80%. Consider enrichment programmes."})
        recommendations.append({"priority": "LOW", "action": "Introduce advanced/enrichment track",
                               "rationale": f"{high_perf} students performing above 80% could benefit from additional challenge."})

    if not recommendations:
        recommendations.append({"priority": "LOW", "action": "Continue current academic support strategies",
                               "rationale": "Current performance indicators are within acceptable range."})

    # ── Render panels ────────────────────────────────────────────────────────
    obs_col, warn_col, rec_col = st.columns(3)

    with obs_col:
        st.subheader(f"📋 Observations ({len(observations)})")
        for obs in observations:
            st.markdown(
                f"""<div style="background:rgba(15,52,96,.5);border-left:4px solid #36e6c2;border-radius:4px;padding:12px;margin-bottom:10px">
                <strong>{obs['icon']} {escape(obs['title'])}</strong><br>
                <span style="color:#a8bad0;font-size:13px">{escape(obs['text'])}</span></div>""",
                unsafe_allow_html=True,
            )

    with warn_col:
        st.subheader(f"⚠️ Risk Signals ({len(warnings)})")
        for warn in warnings:
            color = "#ff6b91" if warn["icon"] == "🚨" else "#ffd166"
            st.markdown(
                f"""<div style="background:rgba(15,52,96,.5);border-left:4px solid {color};border-radius:4px;padding:12px;margin-bottom:10px">
                <strong>{warn['icon']} {escape(warn['title'])}</strong><br>
                <span style="color:#a8bad0;font-size:13px">{escape(warn['text'])}</span></div>""",
                unsafe_allow_html=True,
            )
        if not warnings:
            st.success("No risk signals detected.")

    with rec_col:
        st.subheader(f"💡 Recommendations ({len(recommendations)})")
        priority_colors = {"HIGH": "#ff6b91", "MEDIUM": "#ffd166", "LOW": "#36e6c2"}
        for rec in sorted(recommendations, key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r["priority"], 3)):
            color = priority_colors.get(rec["priority"], "#65c7ff")
            st.markdown(
                f"""<div style="background:rgba(15,52,96,.5);border-left:4px solid {color};border-radius:4px;padding:12px;margin-bottom:10px">
                <span style="color:{color};font-size:10px;font-weight:700">{rec['priority']} PRIORITY</span><br>
                <strong>{escape(rec['action'])}</strong><br>
                <span style="color:#a8bad0;font-size:12px">{escape(rec['rationale'])}</span></div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Phase 8 — Forecasting
# ---------------------------------------------------------------------------

def render_forecasting() -> None:
    """Pass rate and at-risk trajectory forecasts with confidence bands."""
    render_page_header(
        "Forecasting",
        "Time-series forecasts of pass rates, at-risk trajectories, and performance trends using statistical models.",
        "Predictive Analytics",
    )

    if _no_data_banner():
        return

    from models.forecasting import forecast_pass_rate, forecast_at_risk_trajectory

    data, dataset_name = _get_data()

    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Settings")
        forecast_type = st.selectbox(
            "Forecast Type",
            ["pass_rate", "at_risk"],
            format_func=lambda x: "Pass Rate Forecast" if x == "pass_rate" else "At-Risk Trajectory",
            key="fc_type",
        )
        horizon = st.slider("Forecast Horizon (days)", 30, 365, 90, step=30, key="fc_horizon")
        run_btn = st.button("Generate Forecast", type="primary", key="fc_run")

    with col_chart:
        if run_btn or "fc_result" in st.session_state:
            with st.spinner("Computing forecast…"):
                try:
                    if forecast_type == "pass_rate":
                        fc_output = forecast_pass_rate(data, horizon_days=horizon)
                    else:
                        fc_output = forecast_at_risk_trajectory(data, horizon_days=horizon)
                    st.session_state["fc_result"] = fc_output
                    st.session_state["fc_type_last"] = forecast_type
                except Exception as exc:
                    st.error(f"Forecasting failed: {exc}")
                    return

            fc_output = st.session_state.get("fc_result")
            if fc_output is None:
                return

            preds = fc_output.predictions
            point = preds.get("point", [])
            lower = preds.get("lower", [])
            upper = preds.get("upper", [])

            if not point:
                msg = preds.get("message", "Insufficient data for forecasting.")
                st.warning(msg)
                return

            x = list(range(1, len(point) + 1))
            fig = go.Figure()

            if upper and lower:
                fig.add_trace(go.Scatter(
                    x=x + x[::-1], y=upper + lower[::-1],
                    fill="toself", fillcolor="rgba(54,230,194,0.12)",
                    line=dict(color="rgba(255,255,255,0)"), name="95% CI",
                ))
            fig.add_trace(go.Scatter(
                x=x, y=point, mode="lines+markers", name="Forecast",
                line=dict(color="#36e6c2", width=2),
                marker=dict(size=5),
            ))

            title_label = "Pass Rate" if st.session_state.get("fc_type_last") == "pass_rate" else "At-Risk %"
            fig.update_layout(
                title=f"{title_label} Forecast — {horizon}-Day Horizon ({fc_output.method})",
                xaxis_title="Days Ahead",
                yaxis_title=f"{title_label} (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Forecast summary ─────────────────────────────────────────────
            st.caption(
                f"Method: **{fc_output.method}** | "
                f"Data points used: **{fc_output.data_points_used}** | "
                f"Horizon: **{horizon} days** | "
                f"End forecast: **{point[-1]:.1f}%**"
            )


# ---------------------------------------------------------------------------
# Phase 9 — Export Reports
# ---------------------------------------------------------------------------

def render_export_reports() -> None:
    """Generate and download PDF and Excel reports."""
    render_page_header(
        "Export Reports",
        "Generate cohort summary, at-risk analysis, and data quality reports in PDF and Excel formats.",
        "Report Generation",
    )

    data, dataset_name = _get_data()
    has_data = isinstance(data, pd.DataFrame) and not data.empty

    tab1, tab2 = st.tabs(["Excel Reports", "PDF Reports"])

    with tab1:
        _render_excel_reports(data, dataset_name, has_data)

    with tab2:
        _render_pdf_reports(data, dataset_name, has_data)


def _render_excel_reports(data: pd.DataFrame, dataset_name: str, has_data: bool) -> None:
    st.subheader("Excel Report Builder")

    report_type = st.selectbox(
        "Report Type",
        ["Cohort Summary", "At-Risk Students", "Data Quality", "Full Dataset Export"],
        key="xl_report_type",
    )

    if not has_data:
        st.info("Upload a dataset to generate Excel reports.")
        return

    if st.button("Generate Excel Report", type="primary", key="xl_generate"):
        from services.analytics_service import calculate_row_average_scores, get_pass_fail_target
        from utilities.schema_mapping import build_auto_schema_mapping

        schema = build_auto_schema_mapping(data)
        score_cols = [c for c in schema.get("score_columns", []) if c in data.columns]
        row_scores = calculate_row_average_scores(data, score_cols)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            if report_type == "Full Dataset Export":
                data.to_excel(writer, sheet_name="Dataset", index=False)
                if not data.select_dtypes(include="number").empty:
                    data.describe().to_excel(writer, sheet_name="Statistics")

            elif report_type == "Cohort Summary":
                summary = pd.DataFrame({
                    "Metric": ["Total Students", "Pass Rate", "Fail Rate", "Average Score",
                               "At-Risk (< 60%)", "High Performers (≥ 80%)"],
                    "Value": [
                        len(data),
                        f"{(row_scores >= 60).sum() / max(len(row_scores), 1) * 100:.1f}%" if not row_scores.empty else "N/A",
                        f"{(row_scores < 60).sum() / max(len(row_scores), 1) * 100:.1f}%" if not row_scores.empty else "N/A",
                        f"{row_scores.mean():.1f}%" if not row_scores.empty else "N/A",
                        int((row_scores < 60).sum()) if not row_scores.empty else 0,
                        int((row_scores >= 80).sum()) if not row_scores.empty else 0,
                    ],
                })
                summary.to_excel(writer, sheet_name="KPI Summary", index=False)
                data.to_excel(writer, sheet_name="Full Dataset", index=False)
                if not data.select_dtypes(include="number").empty:
                    data.describe().to_excel(writer, sheet_name="Statistics")

            elif report_type == "At-Risk Students":
                if not row_scores.empty:
                    at_risk_df = data.copy()
                    at_risk_df["Average Score"] = row_scores
                    at_risk_df["Risk Level"] = at_risk_df["Average Score"].apply(
                        lambda s: "High Risk" if s < 60 else ("Moderate Risk" if s < 75 else "Low Risk")
                    )
                    at_risk_df.sort_values("Average Score").to_excel(writer, sheet_name="At-Risk Students", index=False)
                    at_risk_df[at_risk_df["Risk Level"] == "High Risk"].to_excel(
                        writer, sheet_name="High Risk Only", index=False
                    )

            elif report_type == "Data Quality":
                quality = pd.DataFrame({
                    "Column": data.columns,
                    "Type": data.dtypes.astype(str).values,
                    "Missing": data.isna().sum().values,
                    "Missing %": (data.isna().mean().values * 100).round(2),
                    "Unique Values": data.nunique(dropna=True).values,
                    "Completeness %": ((1 - data.isna().mean().values) * 100).round(2),
                })
                quality.to_excel(writer, sheet_name="Data Quality", index=False)
                pd.DataFrame({"Metric": ["Total Cells", "Missing Cells", "Duplicate Rows", "Completeness %"],
                              "Value": [data.size, data.isna().sum().sum(), data.duplicated().sum(),
                                       f"{(1 - data.isna().sum().sum() / max(data.size, 1)) * 100:.1f}%"]}
                             ).to_excel(writer, sheet_name="Quality Summary", index=False)

            pd.DataFrame([{"report_type": report_type, "dataset": dataset_name,
                          "rows": len(data), "generated_at": datetime.now().isoformat()}]
                        ).to_excel(writer, sheet_name="Metadata", index=False)

        buf.seek(0)
        file_name = f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.download_button(
            label=f"⬇ Download {report_type} Excel Report",
            data=buf.getvalue(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.success(f"Report ready: **{file_name}**")

        from services.database_service import record_audit_event
        record_audit_event("report_generated", dataset_name, f"Excel: {report_type}", len(data))


def _render_pdf_reports(data: pd.DataFrame, dataset_name: str, has_data: bool) -> None:
    st.subheader("PDF Report Builder")

    if not has_data:
        st.info("Upload a dataset to generate PDF reports.")
        return

    report_type = st.selectbox(
        "Report Type",
        ["Cohort Summary", "At-Risk Analysis"],
        key="pdf_report_type",
    )

    if st.button("Generate PDF Report", type="primary", key="pdf_generate"):
        from services.analytics_service import calculate_row_average_scores
        from utilities.schema_mapping import build_auto_schema_mapping

        schema = build_auto_schema_mapping(data)
        score_cols = [c for c in schema.get("score_columns", []) if c in data.columns]
        row_scores = calculate_row_average_scores(data, score_cols)

        # Build HTML report
        if report_type == "Cohort Summary":
            total = len(data)
            pass_rate = (row_scores >= 60).sum() / max(len(row_scores), 1) * 100 if not row_scores.empty else 0
            avg_score = row_scores.mean() if not row_scores.empty else 0
            at_risk = int((row_scores < 60).sum()) if not row_scores.empty else 0
            html = _build_html_report("Cohort Summary Report", dataset_name, f"""
<h2>Executive Summary</h2>
<table border="1" cellpadding="8" cellspacing="0" style="width:100%;border-collapse:collapse">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Students</td><td>{total:,}</td></tr>
  <tr><td>Pass Rate</td><td>{pass_rate:.1f}%</td></tr>
  <tr><td>Fail Rate</td><td>{100 - pass_rate:.1f}%</td></tr>
  <tr><td>Average Score</td><td>{avg_score:.1f}%</td></tr>
  <tr><td>At-Risk Students</td><td>{at_risk}</td></tr>
  <tr><td>Total Columns</td><td>{len(data.columns)}</td></tr>
  <tr><td>Missing Values</td><td>{data.isna().sum().sum():,}</td></tr>
</table>
<h2>Summary Statistics</h2>
{data.describe().round(2).to_html()}
""")
        else:
            if not row_scores.empty:
                at_risk_df = data.copy()
                at_risk_df["Average Score"] = row_scores.round(1)
                at_risk_df["Risk Level"] = at_risk_df["Average Score"].apply(
                    lambda s: "High Risk" if s < 60 else ("Moderate Risk" if s < 75 else "Low Risk")
                )
                high_risk = at_risk_df[at_risk_df["Risk Level"] == "High Risk"]
                html = _build_html_report("At-Risk Student Report", dataset_name, f"""
<h2>At-Risk Summary</h2>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse">
  <tr><th>Category</th><th>Count</th><th>%</th></tr>
  <tr><td>High Risk (below 60%)</td><td>{len(high_risk)}</td><td>{len(high_risk)/max(len(data),1)*100:.1f}%</td></tr>
  <tr><td>Total Students</td><td>{len(data)}</td><td>100%</td></tr>
</table>
<h2>High Risk Students</h2>
{high_risk.head(100).to_html(index=False)}
""")
            else:
                html = _build_html_report("At-Risk Report", dataset_name, "<p>No score data available.</p>")

        # Try WeasyPrint, fall back to HTML download
        try:
            from weasyprint import HTML as WeasyHTML
            pdf_bytes = WeasyHTML(string=html).write_pdf()
            file_name = f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button("⬇ Download PDF Report", data=pdf_bytes, file_name=file_name, mime="application/pdf")
            st.success(f"PDF ready: **{file_name}**")
        except Exception:
            file_name = f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            st.download_button("⬇ Download HTML Report (PDF unavailable)", data=html.encode(),
                              file_name=file_name, mime="text/html")
            st.info("WeasyPrint not available. Downloading as HTML instead.")

        from services.database_service import record_audit_event
        record_audit_event("report_generated", dataset_name, f"PDF: {report_type}", len(data))


def _build_html_report(title: str, dataset_name: str, body_html: str) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{font-family: Arial, sans-serif; margin: 40px; color: #333; font-size: 13px;}}
  h1 {{background: #0f3460; color: white; padding: 20px; border-radius: 4px;}}
  h2 {{color: #0f3460; border-bottom: 2px solid #36e6c2; padding-bottom: 4px;}}
  table {{border-collapse: collapse; width: 100%; margin-bottom: 20px;}}
  th {{background: #0f3460; color: white; padding: 8px; text-align: left;}}
  td {{padding: 6px 8px; border: 1px solid #ddd;}}
  tr:nth-child(even) {{background: #f8f9fa;}}
  .footer {{color: #999; font-size: 11px; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 10px;}}
</style></head>
<body>
<h1>{escape(title)}</h1>
<p><strong>Dataset:</strong> {escape(dataset_name)} | <strong>Generated:</strong> {generated}</p>
{body_html}
<div class="footer">Generated by Stud-Dash | {generated}</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Phase 10 — Audit Log
# ---------------------------------------------------------------------------

def render_audit_log() -> None:
    """Searchable audit log of uploads, predictions, pipeline events, and report generation."""
    render_page_header(
        "Audit Log",
        "Complete history of dataset uploads, predictions, ETL pipeline runs, and report generation events.",
        "System Audit",
    )

    from services.database_service import get_audit_events, get_dataset_uploads, get_prediction_history

    tab1, tab2, tab3 = st.tabs(["System Events", "Dataset Uploads", "Prediction History"])

    with tab1:
        st.subheader("System Audit Events")
        events = get_audit_events(limit=500)
        if events.empty:
            st.info("No audit events recorded yet. Events are logged automatically as you use the dashboard.")
        else:
            # Filter by event type
            event_types = sorted(events["event_type"].dropna().unique().tolist())
            selected_types = st.multiselect("Filter by event type", event_types, default=[], key="audit_types")
            if selected_types:
                events = events[events["event_type"].isin(selected_types)]

            st.metric("Events Shown", len(events))
            st.dataframe(events, use_container_width=True, hide_index=True)

            # Event type breakdown chart
            if not events.empty:
                breakdown = events["event_type"].value_counts().reset_index()
                breakdown.columns = ["Event Type", "Count"]
                fig = px.bar(breakdown, x="Event Type", y="Count", color="Count",
                            color_continuous_scale="Blues", title="Events by Type")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Dataset Upload History")
        uploads = get_dataset_uploads()
        if uploads.empty:
            st.info("No datasets have been uploaded yet.")
        else:
            st.metric("Total Uploads", len(uploads))
            st.dataframe(uploads, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Prediction History")
        predictions = get_prediction_history(limit=200)
        if predictions.empty:
            st.info("No predictions have been made yet. Use the Prediction page to generate predictions.")
        else:
            st.metric("Total Predictions", len(predictions))

            # Summary metrics
            if "predicted_label" in predictions.columns:
                pass_count = (predictions["predicted_label"] == "Pass").sum()
                fail_count = (predictions["predicted_label"] == "Fail").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Pass Predictions", str(pass_count))
                c2.metric("Fail Predictions", str(fail_count))
                c3.metric("Models Used", str(predictions["model_name"].nunique()) if "model_name" in predictions.columns else "—")

            st.dataframe(predictions, use_container_width=True, hide_index=True)

            if "predicted_label" in predictions.columns:
                fig = px.histogram(predictions, x="risk_level" if "risk_level" in predictions.columns else "predicted_label",
                                  title="Risk Level Distribution",
                                  color_discrete_sequence=["#36e6c2"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
