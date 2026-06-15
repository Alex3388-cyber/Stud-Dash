"""Shared Streamlit shell components and navigation rendering."""

from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from services.analytics_service import calculate_row_average_scores
from services.dataset_service import get_active_kpi_dataset, get_score_columns


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

NAV_ITEMS = [
    # Core dashboard (original 6)
    "Home",
    "Insights",
    "Data Exploration",
    "Prediction",
    "Clustering",
    "Reports",
    # Data Mining & Warehousing (Phase 1–5)
    "Data Warehouse",
    "ETL Monitor",
    "Data Quality",
    "KPI Dashboard",
    "Explainability",
    # Decision Intelligence (Phase 6–10)
    "Advanced Charts",
    "Decision Support",
    "Forecasting",
    "Export Reports",
    "Audit Log",
]

NAV_LABELS = {
    "Home": "Home",
    "Insights": "Insights",
    "Data Exploration": "Data Exploration",
    "Prediction": "Prediction",
    "Clustering": "Clustering",
    "Reports": "Reports",
    "Data Warehouse": "Data Warehouse",
    "ETL Monitor": "ETL Monitor",
    "Data Quality": "Data Quality",
    "KPI Dashboard": "KPI Dashboard",
    "Explainability": "Explainability",
    "Advanced Charts": "Adv. Charts",
    "Decision Support": "Decision Support",
    "Forecasting": "Forecasting",
    "Export Reports": "Export Reports",
    "Audit Log": "Audit Log",
}

NAV_ICONS = {
    "Home": "layout-dashboard",
    "Insights": "activity",
    "Data Exploration": "chart-column",
    "Prediction": "brain-circuit",
    "Clustering": "network",
    "Reports": "file-text",
    # New pages use compact icon keys
    "Data Warehouse": "database",
    "ETL Monitor": "activity",
    "Data Quality": "shield-check",
    "KPI Dashboard": "trending-up",
    "Explainability": "help-circle",
    "Advanced Charts": "bar-chart",
    "Decision Support": "lightbulb",
    "Forecasting": "trending-up",
    "Export Reports": "download",
    "Audit Log": "list",
}

# Compact SVG paths for the new DM/DW pages (shorter than Lucide originals)
COMPACT_ICON_PATHS = {
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "help-circle": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><circle cx="12" cy="17" r=".5"/>',
    "bar-chart": '<rect x="18" y="3" width="4" height="18"/><rect x="10" y="8" width="4" height="13"/><rect x="2" y="13" width="4" height="8"/>',
    "lightbulb": '<path d="M9 21h6m-3-3v-3m0 0a6 6 0 1 0-6-6h0a6 6 0 0 0 6 6"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
}

LUCIDE_ICON_PATHS = {
    "activity": '<path d="M22 12h-4l-3 9-6-18-3 9H2"/>',
    "brain-circuit": (
        '<path d="M12 5a3 3 0 1 0-5.99.34 4 4 0 0 0-2.25 6.74 4 4 0 0 0 2.58 6.58A4 4 0 0 0 12 18"/>'
        '<path d="M12 5a3 3 0 1 1 5.99.34 4 4 0 0 1 2.25 6.74 4 4 0 0 1-2.58 6.58A4 4 0 0 1 12 18"/>'
        '<path d="M12 5v13"/><path d="M8 11h2"/><path d="M14 11h2"/><path d="M9 15h6"/>'
    ),
    "chart-column": (
        '<path d="M3 3v18h18"/>'
        '<rect width="4" height="7" x="7" y="10" rx="1"/>'
        '<rect width="4" height="12" x="15" y="5" rx="1"/>'
        '<path d="M3 14h4"/>'
    ),
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "cloud-upload": (
        '<path d="M12 13v8"/><path d="m8 17 4-4 4 4"/>'
        '<path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>'
        '<path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>'
    ),
    "file-text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>'
    ),
    "graduation-cap": (
        '<path d="M21.42 10.92a1 1 0 0 0-.02-1.84l-8.57-3.9a2 2 0 0 0-1.66 0l-8.57 3.9a1 1 0 0 0 0 1.84l8.57 3.9a2 2 0 0 0 1.66 0z"/>'
        '<path d="M22 10v6"/><path d="M6 12.5V16a6 3 0 0 0 12 0v-3.5"/>'
    ),
    "layout-dashboard": (
        '<rect width="7" height="9" x="3" y="3" rx="1"/>'
        '<rect width="7" height="5" x="14" y="3" rx="1"/>'
        '<rect width="7" height="9" x="14" y="12" rx="1"/>'
        '<rect width="7" height="5" x="3" y="16" rx="1"/>'
    ),
    "network": (
        '<rect x="9" y="2" width="6" height="6" rx="1"/>'
        '<rect x="2" y="16" width="6" height="6" rx="1"/>'
        '<rect x="16" y="16" width="6" height="6" rx="1"/>'
        '<path d="M12 8v4"/><path d="M5 16v-2a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v2"/>'
    ),
    "shield-check": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
}


def extract_numeric_display_value(value: str) -> tuple[float | None, str, str]:
    """Extract a numeric value plus prefix and suffix for count-up display."""
    clean_value = value.strip()
    if clean_value.upper() == "N/A":
        return None, "", clean_value

    prefix = ""
    suffix = ""
    numeric_part = clean_value.replace(",", "")

    if numeric_part.startswith("$"):
        prefix = "$"
        numeric_part = numeric_part[1:]

    if numeric_part.endswith("%"):
        suffix = "%"
        numeric_part = numeric_part[:-1]

    try:
        return float(numeric_part), prefix, suffix
    except ValueError:
        return None, "", clean_value


def load_custom_css() -> None:
    """Load project CSS and inject theme-switching JavaScript."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{get_custom_css_text(str(css_path))}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        (function () {
            var t = localStorage.getItem('sd-theme') || 'dark';
            document.documentElement.setAttribute('data-theme', t);

            window.toggleTheme = function () {
                var c = document.documentElement.getAttribute('data-theme') || 'dark';
                var n = c === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', n);
                localStorage.setItem('sd-theme', n);
                var icon = document.querySelector('.theme-icon');
                var lbl  = document.querySelector('.theme-toggle-label');
                if (icon) icon.textContent = n === 'dark' ? '☀' : '☾';
                if (lbl)  lbl.textContent  = n === 'dark' ? 'Light mode' : 'Dark mode';
            };

            setTimeout(function () {
                var theme = document.documentElement.getAttribute('data-theme') || 'dark';
                var icon  = document.querySelector('.theme-icon');
                var lbl   = document.querySelector('.theme-toggle-label');
                if (icon) icon.textContent = theme === 'dark' ? '☾' : '☀';
                if (lbl)  lbl.textContent  = theme === 'dark' ? 'Light mode' : 'Dark mode';
            }, 250);
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_custom_css_text(css_path: str) -> str:
    """Cache CSS text so the stylesheet is not re-read on every rerun."""
    return Path(css_path).read_text(encoding="utf-8")


def lucide_icon(icon_name: str, class_name: str = "lucide-icon") -> str:
    """Return inline SVG markup for a Lucide-style icon (compact or full)."""
    paths = (
        COMPACT_ICON_PATHS.get(icon_name)
        or LUCIDE_ICON_PATHS.get(icon_name)
        or LUCIDE_ICON_PATHS["activity"]
    )
    return (
        f'<svg class="{class_name}" xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


def get_uploaded_score_distribution_bars(limit: int = 5) -> str:
    """Build small console bars from the active dataset score distribution."""
    data, _dataset_name, _source_label = get_active_kpi_dataset()
    if not isinstance(data, pd.DataFrame) or data.empty:
        heights = [8] * limit
    else:
        score_columns = get_score_columns(data)
        scores = calculate_row_average_scores(data, score_columns)
        clean_scores = scores.dropna()
        if clean_scores.empty:
            heights = [8] * limit
        else:
            quantiles = clean_scores.clip(lower=0, upper=100).quantile(
                [index / max(limit - 1, 1) for index in range(limit)]
            )
            heights = [max(10, round(float(value))) for value in quantiles]
    return "".join(f'<span style="height: {height}%"></span>' for height in heights)


def render_page_header(title: str, subtitle: str, label: str = "Academic Analytics") -> None:
    """Render a consistent header for dashboard sections."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-kicker">
                <span class="kicker-dot"></span>
                {escape(label)}
            </div>
            <h1 class="page-title">{escape(title)}</h1>
            <p class="page-subtitle">{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, note: str, icon: str = "") -> None:
    """Render a compact metric card."""
    icon_markup = f'<div class="metric-icon">{icon}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-top">
                <span>{title}</span>
                {icon_markup}
            </div>
            <strong>{value}</strong>
            <small>{note}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str, items: list[str] | None = None) -> None:
    """Render a reusable dashboard panel."""
    list_items = ""
    if items:
        list_items = "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    st.markdown(
        f"""
        <section class="dashboard-panel">
            <h3>{title}</h3>
            <p>{body}</p>
            {list_items}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_placeholder(title: str, subtitle: str, label: str = "Layout placeholder") -> None:
    """Render a styled placeholder panel."""
    st.markdown(
        f"""
        <div class="placeholder-panel">
            <span>{label}</span>
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render the custom sidebar navigation and return the selected page."""
    query_params = st.query_params
    requested_page = query_params.get("page", NAV_ITEMS[0])
    selected_page = requested_page if requested_page in NAV_ITEMS else NAV_ITEMS[0]

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="brand-mark">{lucide_icon("graduation-cap", "brand-icon")}</div>
            <div>
                <strong>Student PP</strong>
                <span>AI academic analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Split navigation into 3 small blocks to stay under Streamlit's markdown
    # character limit for sidebar content with inline SVG icons.
    def _nav_block(items: list[str]) -> str:
        markup = ['<nav class="sidebar-nav" aria-label="Dashboard navigation">']
        for page in items:
            active_class = " active" if page == selected_page else ""
            page_url = f"?page={quote(page)}"
            markup.append(
                f'<a class="sidebar-nav-item{active_class}" href="{page_url}" target="_self">'
                f'<span class="sidebar-nav-icon">{lucide_icon(NAV_ICONS[page])}</span>'
                f'<span class="sidebar-nav-label">{escape(NAV_LABELS[page])}</span>'
                f'<span class="sidebar-nav-arrow">{lucide_icon("chevron-right", "sidebar-arrow-icon")}</span>'
                "</a>"
            )
        markup.append("</nav>")
        return "".join(markup)

    # Block 1: Core dashboard pages (original 6)
    core_pages = [p for p in NAV_ITEMS if p in {
        "Home", "Insights", "Data Exploration", "Prediction", "Clustering", "Reports"
    }]
    st.sidebar.markdown(
        '<div class="sidebar-section-label">Core Dashboard</div>' + _nav_block(core_pages),
        unsafe_allow_html=True,
    )

    # Block 2: Data Mining / Warehousing pages
    dm_pages = [p for p in NAV_ITEMS if p in {
        "Data Warehouse", "ETL Monitor", "Data Quality", "KPI Dashboard", "Explainability"
    }]
    st.sidebar.markdown(
        '<div class="sidebar-section-label">Data Mining &amp; Warehouse</div>' + _nav_block(dm_pages),
        unsafe_allow_html=True,
    )

    # Block 3: Decision Intelligence pages
    di_pages = [p for p in NAV_ITEMS if p in {
        "Advanced Charts", "Decision Support", "Forecasting", "Export Reports", "Audit Log"
    }]
    st.sidebar.markdown(
        '<div class="sidebar-section-label">Decision Intelligence</div>' + _nav_block(di_pages),
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
        <div class="sidebar-status-card">
            <div>{lucide_icon("shield-check", "sidebar-status-icon")}<strong>DM/DW Platform Online</strong></div>
            <p>16 modules: warehouse, ETL, quality, KPIs, XAI, forecasting, reports, and audit active.</p>
            <div class="sidebar-data-pill">{lucide_icon("database", "sidebar-pill-icon")} SQLite connected</div>
        </div>
        <button class="theme-toggle-btn" onclick="toggleTheme()" aria-label="Toggle colour theme">
            <span class="theme-icon">☾</span>
            <span class="theme-toggle-label">Light mode</span>
        </button>
        """,
        unsafe_allow_html=True,
    )
    return selected_page
