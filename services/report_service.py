"""Report generation service — produces PDF and Excel reports."""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPORTS_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "output"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "reports" / "templates"


def _ensure_output_dir() -> Path:
    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_OUTPUT_DIR


def _render_template(title: str, subtitle: str, dataset_name: str, body_html: str) -> str:
    template_path = TEMPLATES_DIR / "base.html"
    if template_path.exists():
        try:
            from jinja2 import Template
            template = Template(template_path.read_text(encoding="utf-8"))
            return template.render(
                title=title,
                subtitle=subtitle,
                dataset_name=dataset_name,
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                body=body_html,
            )
        except Exception:
            pass
    return f"<html><body><h1>{title}</h1>{body_html}</body></html>"


def _df_to_html_table(data: pd.DataFrame, max_rows: int = 200) -> str:
    if data is None or data.empty:
        return "<p><em>No data available.</em></p>"
    subset = data.head(max_rows)
    return subset.to_html(index=False, classes="report-table", border=0)


def _build_cohort_summary_body(data: pd.DataFrame | None) -> str:
    if data is None or data.empty:
        return "<p>No dataset provided for this report.</p>"
    rows, cols = len(data), len(data.columns)
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    body = f"""
    <div class="section">
      <h2>Dataset Overview</h2>
      <div class="kpi-grid">
        <div class="kpi-card"><div class="label">Total Records</div><div class="value">{rows:,}</div></div>
        <div class="kpi-card"><div class="label">Columns</div><div class="value">{cols}</div></div>
        <div class="kpi-card"><div class="label">Numeric Fields</div><div class="value">{len(numeric_cols)}</div></div>
      </div>
    </div>
    <div class="section">
      <h2>Summary Statistics</h2>
      {_df_to_html_table(data.describe(include="all").T.reset_index().rename(columns={"index": "Column"}))}
    </div>
    <div class="section">
      <h2>Missing Values</h2>
      {_df_to_html_table(pd.DataFrame({"Column": data.columns, "Missing": data.isna().sum().values, "% Missing": (data.isna().mean().values * 100).round(2)}).sort_values("Missing", ascending=False))}
    </div>
    """
    return body


def _build_at_risk_body(data: pd.DataFrame | None) -> str:
    if data is None or data.empty:
        return "<p>No dataset provided.</p>"
    from utilities.schema_mapping import build_auto_schema_mapping
    from services.analytics_service import calculate_row_average_scores
    schema = build_auto_schema_mapping(data)
    score_cols = [c for c in schema.get("score_columns", []) if c in data.columns]
    row_scores = calculate_row_average_scores(data, score_cols)
    if row_scores.empty:
        return "<p>No score columns detected for at-risk analysis.</p>"
    at_risk_data = data.copy()
    at_risk_data["Average Score"] = row_scores
    at_risk_data["Risk Level"] = at_risk_data["Average Score"].apply(
        lambda s: "High Risk" if s < 60 else ("Moderate Risk" if s < 75 else "Low Risk")
    )
    at_risk = at_risk_data[at_risk_data["Risk Level"] == "High Risk"].reset_index(drop=True)
    total = len(data)
    at_risk_count = len(at_risk)
    body = f"""
    <div class="section">
      <h2>At-Risk Summary</h2>
      <div class="kpi-grid">
        <div class="kpi-card"><div class="label">Total Students</div><div class="value">{total:,}</div></div>
        <div class="kpi-card"><div class="label">At-Risk (below 60%)</div><div class="value">{at_risk_count:,}</div></div>
        <div class="kpi-card"><div class="label">At-Risk Rate</div><div class="value">{at_risk_count/max(total,1)*100:.1f}%</div></div>
      </div>
    </div>
    <div class="section">
      <h2>At-Risk Student Records</h2>
      {_df_to_html_table(at_risk.head(100))}
    </div>
    """
    return body


def _to_pdf(html: str, output_path: Path) -> int:
    """Render HTML to PDF using WeasyPrint. Falls back to saving HTML if unavailable."""
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(output_path))
    except Exception:
        html_path = output_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        output_path = html_path
    return int(output_path.stat().st_size)


def _to_excel(data: pd.DataFrame | None, output_path: Path, report_type: str) -> int:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        if data is not None and not data.empty:
            data.to_excel(writer, sheet_name="Data", index=False)
            if data.select_dtypes(include="number").columns.any():
                data.describe().to_excel(writer, sheet_name="Summary Statistics")
        pd.DataFrame([{"report_type": report_type, "generated_at": datetime.now().isoformat()}]).to_excel(
            writer, sheet_name="Metadata", index=False
        )
    output_path.write_bytes(out.getvalue())
    return int(output_path.stat().st_size)


REPORT_BUILDERS = {
    "cohort_summary": ("Cohort Summary Report", "Overview of student dataset, statistics, and data quality", _build_cohort_summary_body),
    "at_risk": ("At-Risk Student Report", "Students below the 60% performance threshold", _build_at_risk_body),
}


def build_report(
    report_id: int,
    report_type: str,
    format: str,
    data: pd.DataFrame | None,
    options: dict[str, Any],
) -> tuple[str, int]:
    output_dir = _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_type = report_type.replace(" ", "_").lower()
    dataset_name = options.get("dataset_name", "Uploaded dataset")

    title, subtitle, body_builder = REPORT_BUILDERS.get(
        report_type,
        (report_type.replace("_", " ").title(), "Generated report", _build_cohort_summary_body),
    )

    if format == "xlsx":
        output_path = output_dir / f"{safe_type}_{report_id}_{timestamp}.xlsx"
        size = _to_excel(data, output_path, report_type)
    else:
        body_html = body_builder(data)
        html = _render_template(title, subtitle, dataset_name, body_html)
        output_path = output_dir / f"{safe_type}_{report_id}_{timestamp}.pdf"
        size = _to_pdf(html, output_path)

    return str(output_path), size
