"""Shared premium Plotly styling for the dashboard.

The functions in this file only change chart presentation: colors,
backgrounds, spacing, legends, hover labels, and trace styling.
"""

from __future__ import annotations


PREMIUM_COLORWAY = [
    "#36e6c2",
    "#65c7ff",
    "#ffd166",
    "#9b7cff",
    "#ff6b91",
    "#48e69b",
    "#f78c6b",
    "#b6e2ff",
]

PREMIUM_CONTINUOUS_SCALE = [
    [0.0, "#071426"],
    [0.18, "#12365a"],
    [0.38, "#246da6"],
    [0.62, "#36e6c2"],
    [0.82, "#ffd166"],
    [1.0, "#ff6b91"],
]

PREMIUM_DIVERGING_SCALE = [
    [0.0, "#ff776d"],
    [0.22, "#9b7cff"],
    [0.48, "#12233f"],
    [0.72, "#65c7ff"],
    [1.0, "#36e6c2"],
]


def apply_premium_chart_theme(figure, height: int | None = None):
    """Apply premium dark analytics styling without changing chart data."""
    layout_updates = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(8,18,35,0.22)",
        "colorway": PREMIUM_COLORWAY,
        "font": {"family": "Segoe UI, Arial, sans-serif", "color": "#edf7ff", "size": 13},
        "title": {
            "font": {"color": "#edf7ff", "size": 17},
            "x": 0.02,
            "xanchor": "left",
            "pad": {"t": 6, "b": 12},
        },
        "margin": {"l": 28, "r": 18, "t": 58, "b": 34},
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "rgba(12,24,42,0.96)",
            "bordercolor": "#36e6c2",
            "font": {"family": "Segoe UI, Arial, sans-serif", "color": "#edf7ff", "size": 12},
            "align": "left",
            "namelength": -1,
        },
        "legend": {
            "title": {"text": ""},
            "bgcolor": "rgba(12,24,42,0.72)",
            "bordercolor": "rgba(182,226,255,0.22)",
            "borderwidth": 1,
            "font": {"color": "#edf7ff", "size": 12},
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "bottom",
            "itemsizing": "constant",
            "itemclick": "toggleothers",
            "itemdoubleclick": "toggle",
        },
        "modebar": {
            "bgcolor": "rgba(12,24,42,0.72)",
            "color": "#a8bad0",
            "activecolor": "#36e6c2",
        },
        "dragmode": "pan",
        "uirevision": "premium-dashboard",
        "transition": {"duration": 180, "easing": "cubic-in-out"},
    }
    if height is not None:
        layout_updates["height"] = height

    figure.update_layout(**layout_updates)
    figure.update_xaxes(
        automargin=True,
        gridcolor="rgba(182,226,255,0.10)",
        linecolor="rgba(182,226,255,0.20)",
        mirror=False,
        showgrid=True,
        showline=True,
        showspikes=True,
        spikecolor="rgba(54,230,194,0.48)",
        spikethickness=1.2,
        spikesnap="cursor",
        tickfont={"color": "#a8bad0", "size": 12},
        title_font={"color": "#edf7ff", "size": 13},
        zerolinecolor="rgba(182,226,255,0.16)",
    )
    figure.update_yaxes(
        automargin=True,
        gridcolor="rgba(182,226,255,0.10)",
        linecolor="rgba(182,226,255,0.20)",
        mirror=False,
        showgrid=True,
        showline=True,
        showspikes=True,
        spikecolor="rgba(54,230,194,0.48)",
        spikethickness=1.2,
        spikesnap="cursor",
        tickfont={"color": "#a8bad0", "size": 12},
        title_font={"color": "#edf7ff", "size": 13},
        zerolinecolor="rgba(182,226,255,0.16)",
    )

    if "coloraxis" in figure.layout:
        figure.update_layout(
            coloraxis={
                "colorscale": PREMIUM_CONTINUOUS_SCALE,
                "colorbar": {
                    "bgcolor": "rgba(12,24,42,0.72)",
                    "bordercolor": "rgba(182,226,255,0.22)",
                    "borderwidth": 1,
                    "tickfont": {"color": "#a8bad0"},
                    "title": {"font": {"color": "#edf7ff"}},
                    "thickness": 14,
                },
            }
        )

    style_premium_traces(figure)
    return figure


def style_premium_traces(figure) -> None:
    """Polish common Plotly trace types used by the dashboard."""
    figure.update_traces(
        marker_line_color="rgba(237,247,255,0.36)",
        marker_line_width=0.7,
        opacity=0.92,
        hoverlabel={"bordercolor": "#65c7ff"},
        selector={"type": "bar"},
    )
    figure.update_traces(
        marker_line_color="rgba(237,247,255,0.78)",
        marker_line_width=0.9,
        marker_size=11,
        opacity=0.9,
        hoverlabel={"bordercolor": "#36e6c2"},
        selector={"type": "scatter"},
    )
    figure.update_traces(
        line_width=3,
        marker_size=8,
        marker_line_color="rgba(237,247,255,0.78)",
        marker_line_width=0.9,
        line_shape="spline",
        selector={"type": "scatter", "mode": "lines+markers"},
    )
    figure.update_traces(
        marker_line_color="rgba(237,247,255,0.34)",
        marker_line_width=0.6,
        opacity=0.88,
        hoverlabel={"bordercolor": "#ffd166"},
        selector={"type": "histogram"},
    )
    figure.update_traces(
        marker_line_color="rgba(5,9,20,0.74)",
        marker_line_width=2,
        textfont={"color": "#edf7ff"},
        sort=False,
        selector={"type": "pie"},
    )
    figure.update_traces(
        xgap=2,
        ygap=2,
        colorbar={"outlinecolor": "rgba(182,226,255,0.22)", "outlinewidth": 1},
        selector={"type": "heatmap"},
    )
    figure.update_traces(
        marker_color="#65c7ff",
        line_color="#36e6c2",
        fillcolor="rgba(101,199,255,0.18)",
        selector={"type": "box"},
    )
