"""New EDW dashboard page renderers — Phases 3–9 additions."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.shell import render_page_header, render_panel


# ---------------------------------------------------------------------------
# Helper — get DB session for Streamlit pages
# ---------------------------------------------------------------------------

def _get_db_and_dataset():
    """Return (db_session, dataset_id, dataframe) from the most recent upload."""
    try:
        from database.pg_connection import get_db as _get_db
        from database.pg_operations import get_latest_dataset_upload, load_dataset_as_dataframe
        with _get_db() as db:
            upload = get_latest_dataset_upload(db)
            if not upload:
                return None, None, None
            data = load_dataset_as_dataframe(db, upload.id)
            return db, upload.id, data
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Data Warehouse page
# ---------------------------------------------------------------------------

def render_data_warehouse() -> None:
    render_page_header("Data Warehouse", "Monitor ETL jobs, dataset registry, and storage health.", "Warehouse")

    try:
        from database.pg_connection import get_db, init_db
        from database.pg_operations import list_dataset_uploads, list_etl_jobs
        init_db()

        with get_db() as db:
            uploads = list_dataset_uploads(db, limit=20)
            jobs = list_etl_jobs(db, limit=20)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Registry")
            if not uploads:
                st.info("No datasets ingested yet. Use the Upload page or POST /etl/upload.")
            else:
                rows = [{
                    "ID": u.id,
                    "File": u.file_name,
                    "Rows": u.row_count,
                    "Columns": u.column_count,
                    "Quality": f"{u.quality_score:.0f}%" if u.quality_score else "N/A",
                    "Status": u.status,
                    "Uploaded": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "",
                } for u in uploads]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with col2:
            st.subheader("ETL Job History")
            if not jobs:
                st.info("No ETL jobs recorded yet.")
            else:
                job_rows = [{
                    "ID": j.id,
                    "Status": j.status,
                    "Stage": j.stage,
                    "Loaded": j.rows_loaded,
                    "Rejected": j.rows_rejected,
                    "Duration": f"{j.duration_seconds:.1f}s" if j.duration_seconds else "—",
                    "Started": j.started_at.strftime("%H:%M:%S") if j.started_at else "",
                } for j in jobs]
                st.dataframe(pd.DataFrame(job_rows), use_container_width=True)

    except Exception as exc:
        st.error(f"Warehouse connection error: {exc}")
        st.info("Ensure the database is initialised and DATABASE_URL is set in .env")


# ---------------------------------------------------------------------------
# Data Quality page
# ---------------------------------------------------------------------------

def render_data_quality() -> None:
    render_page_header("Data Quality", "Completeness, outliers, schema conformance, and quality scores.", "Quality")

    try:
        from database.pg_connection import get_db
        from database.pg_operations import get_latest_dataset_upload, load_dataset_as_dataframe
        from etl.validator import validate

        with get_db() as db:
            upload = get_latest_dataset_upload(db)
            if not upload:
                st.info("No dataset available. Upload a file first.")
                return

            data = load_dataset_as_dataframe(db, upload.id)

        if data is None or data.empty:
            st.info("Dataset is empty.")
            return

        report = validate(data)

        # Score card
        score_col, comp_col, dup_col, out_col = st.columns(4)
        score_col.metric("Overall Quality Score", f"{report.overall_score:.0f} / 100")
        comp_col.metric("Completeness", f"{report.completeness_score:.1f}%")
        dup_col.metric("Duplicate Rate", f"{report.duplicate_rate:.1%}")
        out_col.metric("Outlier Columns", len(report.outlier_flags))

        if report.warnings:
            st.warning("**Quality Warnings**")
            for w in report.warnings:
                st.markdown(f"- {escape(w)}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Column Completeness")
            completeness = pd.DataFrame({
                "Column": data.columns,
                "Non-Null %": ((1 - data.isna().mean()) * 100).round(1).values,
            }).sort_values("Non-Null %")
            fig = px.bar(completeness, x="Non-Null %", y="Column", orientation="h",
                         color="Non-Null %", color_continuous_scale="RdYlGn", range_color=[0, 100])
            fig.update_layout(height=max(300, len(data.columns) * 22), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Outlier Distribution")
            if report.outlier_flags:
                outl_df = pd.DataFrame([{"Column": c, "Outlier Rows": n} for c, n in report.outlier_flags.items()])
                fig2 = px.bar(outl_df, x="Column", y="Outlier Rows", color="Outlier Rows",
                              color_continuous_scale="Reds")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.success("No statistical outliers detected (z-score > 3).")

    except Exception as exc:
        st.error(f"Quality check failed: {exc}")


# ---------------------------------------------------------------------------
# KPI Dashboard page
# ---------------------------------------------------------------------------

def render_kpi_dashboard() -> None:
    render_page_header("KPI Dashboard", "Materialised key performance indicators from the warehouse.", "KPIs")

    try:
        from database.pg_connection import get_db
        from database.pg_operations import get_latest_dataset_upload
        from services.kpi_engine import get_kpis_for_dataset

        with get_db() as db:
            upload = get_latest_dataset_upload(db)
            if not upload:
                st.info("No dataset in warehouse. Upload a file first.")
                return
            kpis = get_kpis_for_dataset(db, upload.id)

        if not kpis:
            st.info("KPIs are being computed. Refresh in a moment.")
            return

        kpi_dict = {k["name"]: k for k in kpis}

        # Top metric row
        cols = st.columns(4)
        for idx, name in enumerate(["total_students", "pass_rate", "fail_rate", "average_score"]):
            kpi = kpi_dict.get(name, {})
            cols[idx].metric(name.replace("_", " ").title(), kpi.get("label", "N/A"))

        cols2 = st.columns(3)
        for idx, name in enumerate(["average_attendance", "at_risk_count", "high_performers_count"]):
            kpi = kpi_dict.get(name, {})
            cols2[idx].metric(name.replace("_", " ").title(), kpi.get("label", "N/A"))

        # KPI table
        st.subheader("All Materialised KPIs")
        kpi_df = pd.DataFrame([{"KPI": k["name"].replace("_", " ").title(), "Value": k.get("label", "N/A"), "Group": k.get("group", "")} for k in kpis])
        st.dataframe(kpi_df, use_container_width=True)

    except Exception as exc:
        st.error(f"KPI load failed: {exc}")


# ---------------------------------------------------------------------------
# Forecasting page
# ---------------------------------------------------------------------------

def render_forecasting() -> None:
    render_page_header("Forecasting", "Projected pass rates, at-risk trajectories, and score trends.", "Forecasting")

    try:
        from database.pg_connection import get_db
        from database.pg_operations import get_latest_dataset_upload, load_dataset_as_dataframe
        from services.forecasting_service import generate_forecast

        with get_db() as db:
            upload = get_latest_dataset_upload(db)
            if not upload:
                st.info("No dataset available. Upload a file first.")
                return
            data = load_dataset_as_dataframe(db, upload.id)

        if data is None or data.empty:
            st.info("Dataset is empty.")
            return

        col1, col2 = st.columns([2, 1])
        with col2:
            forecast_type = st.selectbox("Forecast Type", ["pass_rate", "at_risk"], format_func=lambda x: x.replace("_", " ").title())
            horizon = st.slider("Horizon (days)", 30, 365, 90, step=30)
            run_btn = st.button("Generate Forecast", type="primary")

        with col1:
            if run_btn or True:
                with st.spinner("Computing forecast..."):
                    result = generate_forecast(data, forecast_type=forecast_type, horizon_days=horizon)

                preds = result.get("predictions", {})
                point = preds.get("point", [])
                lower = preds.get("lower", [])
                upper = preds.get("upper", [])

                if not point:
                    st.warning(preds.get("message", "Insufficient data for forecasting."))
                    return

                x = list(range(1, len(point) + 1))
                fig = go.Figure()
                if upper and lower:
                    fig.add_trace(go.Scatter(x=x + x[::-1], y=upper + lower[::-1], fill="toself",
                                             fillcolor="rgba(15,52,96,0.12)", line=dict(color="rgba(255,255,255,0)"),
                                             name="95% CI"))
                fig.add_trace(go.Scatter(x=x, y=point, mode="lines+markers", name="Forecast",
                                         line=dict(color="#0f3460", width=2)))
                fig.update_layout(
                    title=f"{forecast_type.replace('_', ' ').title()} Forecast ({result['method']})",
                    xaxis_title="Days Ahead",
                    yaxis_title="% Value",
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Method: {result['method']} | Data points used: {result['data_points_used']}")

    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")


# ---------------------------------------------------------------------------
# Explainability page
# ---------------------------------------------------------------------------

def render_explainability() -> None:
    render_page_header("Explainability", "SHAP values, feature contributions, and counterfactual analysis.", "XAI")

    try:
        from database.pg_connection import get_db
        from database.pg_operations import get_latest_dataset_upload, load_dataset_as_dataframe, list_prediction_events
        from models.explainability import build_counterfactual, explain_prediction
        from models.student_prediction import train_logistic_prediction_model

        with get_db() as db:
            upload = get_latest_dataset_upload(db)
            if not upload:
                st.info("No dataset available.")
                return
            data = load_dataset_as_dataframe(db, upload.id)
            recent_predictions = list_prediction_events(db, limit=10)

        if data is None or data.empty:
            st.info("Dataset is empty.")
            return

        bundle = train_logistic_prediction_model(data)

        st.subheader("Single Student Explanation")
        col1, col2, col3, col4, col5 = st.columns(5)
        study_time = col1.number_input("Study Time", 0.0, 20.0, 2.0, key="xai_study")
        absences = col2.number_input("Absences", 0.0, 100.0, 5.0, key="xai_abs")
        failures = col3.number_input("Failures", 0.0, 10.0, 0.0, key="xai_fail")
        g1 = col4.number_input("Grade G1", 0.0, 20.0, 10.0, key="xai_g1")
        g2 = col5.number_input("Grade G2", 0.0, 20.0, 10.0, key="xai_g2")

        if st.button("Explain This Prediction", type="primary"):
            import pandas as _pd
            features = _pd.DataFrame([{"study_time": study_time, "absences": absences, "failures": failures,
                                        "previous_grade_1": g1, "previous_grade_2": g2}])

            from models.student_prediction import predict_student_performance
            result = predict_student_performance(bundle, study_time=study_time, absences=absences,
                                                  failures=failures, previous_grade_1=g1, previous_grade_2=g2)

            pred_col, risk_col = st.columns(2)
            pred_col.metric("Prediction", result.predicted_label)
            risk_col.metric("Risk Level", result.risk_level)
            st.info(f"**Recommendation:** {result.recommendation}")

            shap_data = explain_prediction(bundle.model, features)
            if shap_data and shap_data.get("top_contributors"):
                st.subheader("Feature Contributions")
                contributors = shap_data["top_contributors"]
                contrib_df = pd.DataFrame(contributors)
                fig = px.bar(contrib_df, x="shap_value", y="feature", orientation="h",
                             color="shap_value", color_continuous_scale="RdBu_r",
                             title="SHAP Feature Contributions (positive = pushes toward Pass)",
                             labels={"shap_value": "SHAP Value", "feature": "Feature"})
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Method: {shap_data.get('method', 'unknown')}")

            cf = build_counterfactual(bundle.model, features)
            if cf and "message" not in cf[0]:
                st.subheader("Counterfactual: What Would Change the Prediction?")
                st.dataframe(pd.DataFrame(cf), use_container_width=True)

    except Exception as exc:
        st.error(f"Explainability failed: {exc}")


# ---------------------------------------------------------------------------
# Audit Log page
# ---------------------------------------------------------------------------

def render_audit_log() -> None:
    render_page_header("Audit Log", "Complete action history for uploads, predictions, and training runs.", "Audit")

    try:
        from database.pg_connection import get_db
        from database.pg_operations import audit_logs_as_dataframe

        with get_db() as db:
            df = audit_logs_as_dataframe(db, limit=500)

        if df.empty:
            st.info("No audit events recorded yet.")
            return

        action_filter = st.multiselect("Filter by action", sorted(df["action"].unique().tolist()), default=[])
        if action_filter:
            df = df[df["action"].isin(action_filter)]

        st.metric("Total Events", len(df))
        st.dataframe(df, use_container_width=True)

        if not df.empty and "action" in df.columns:
            action_counts = df["action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Count"]
            fig = px.bar(action_counts, x="Action", y="Count", color="Count",
                         color_continuous_scale="Blues", title="Events by Action Type")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Audit log unavailable: {exc}")


# ---------------------------------------------------------------------------
# ETL Monitor page  (DB-backed)
# ---------------------------------------------------------------------------

def render_etl_monitor() -> None:
    render_page_header(
        "ETL Monitor",
        "Track extraction, transformation, and loading stages with timings, row counts, and error details.",
        "Pipeline Monitor",
    )

    try:
        from database.pg_connection import get_db, init_db
        from database.pg_operations import list_etl_jobs, list_dataset_uploads
        init_db()

        with get_db() as db:
            jobs = list_etl_jobs(db, limit=100)
            uploads = list_dataset_uploads(db, limit=20)

        if not jobs:
            st.info(
                "No ETL jobs recorded yet. "
                "Upload a dataset to populate the pipeline monitor."
            )
            _render_pipeline_status_key()
            return

        total = len(jobs)
        completed = sum(1 for j in jobs if j.status == "completed")
        failed = sum(1 for j in jobs if j.status == "failed")
        avg_dur = (
            sum(j.duration_seconds or 0 for j in jobs if j.duration_seconds) /
            max(1, sum(1 for j in jobs if j.duration_seconds))
        )

        cols = st.columns(4)
        cols[0].metric("Total Pipeline Runs", str(total))
        cols[1].metric("Completed", str(completed))
        cols[2].metric("Failed", str(failed), f"-{failed}" if failed else None)
        cols[3].metric("Avg Duration", f"{avg_dur:.1f}s" if avg_dur else "—")

        st.divider()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Pipeline Job History")
            rows = [
                {
                    "ID": j.id,
                    "Status": j.status,
                    "Stage": j.stage,
                    "Rows In": j.rows_extracted,
                    "Rows Out": j.rows_loaded,
                    "Rejected": j.rows_rejected,
                    "Duration (s)": round(j.duration_seconds, 2) if j.duration_seconds else None,
                    "Started": j.started_at.strftime("%Y-%m-%d %H:%M") if j.started_at else None,
                }
                for j in jobs
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Stage Distribution")
            stage_counts = {}
            for j in jobs:
                stage_counts[j.stage] = stage_counts.get(j.stage, 0) + 1
            if stage_counts:
                fig = px.pie(
                    names=list(stage_counts.keys()),
                    values=list(stage_counts.values()),
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Records Processed Over Time")
        sorted_jobs = sorted(jobs, key=lambda j: j.started_at or j.id)
        times = [j.started_at.strftime("%Y-%m-%d %H:%M") if j.started_at else str(j.id) for j in sorted_jobs]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=times, y=[j.rows_extracted for j in sorted_jobs], name="Rows Extracted", marker_color="#38bdf8"))
        fig2.add_trace(go.Bar(x=times, y=[j.rows_loaded for j in sorted_jobs], name="Rows Loaded", marker_color="#2dd4bf"))
        fig2.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="group", xaxis_title="Run Time", yaxis_title="Record Count",
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as exc:
        st.error(f"ETL monitor unavailable: {exc}")


def _render_pipeline_status_key() -> None:
    st.markdown(
        """
        <div style="background:rgba(15,52,96,.4);border:1px solid #1e4d7e;border-radius:8px;padding:16px;margin-top:16px">
          <strong>Pipeline Stages Tracked</strong>
          <ul style="margin:8px 0 0;color:#a8bad0;font-size:13px">
            <li><b>Extract</b> — Read raw file bytes and parse CSV/Excel</li>
            <li><b>Validate</b> — Schema check, type detection, quality scoring</li>
            <li><b>Transform</b> — Dedup, null fill, scaling, encoding</li>
            <li><b>Load</b> — Persist rows to warehouse</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Advanced Charts, Decision Support, Export Reports — re-export from dm_pages
# (These pages are dataset-centric, not DB-layer-backed, so no migration needed)
# ---------------------------------------------------------------------------

from ui.dm_pages import (
    render_advanced_charts,
    render_decision_support,
    render_export_reports,
)
