"""Interactive Plotly chart builders with unified theme and zero-baseline axes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.insights import SEGMENT_COLORS, SEGMENT_ORDER

PALETTE = {
    "primary": "#2563EB",
    "secondary": "#0D9488",
    "accent": "#D97706",
    "danger": "#DC2626",
    "muted": "#94A3B8",
    "grid": "rgba(148, 163, 184, 0.22)",
    "q4_band": "rgba(217, 119, 6, 0.10)",
    "q4_border": "rgba(217, 119, 6, 0.35)",
}


def style_figure(
    fig: go.Figure,
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
) -> go.Figure:
    """Apply consistent theme-adaptive layout compatible with light and dark modes."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=margin or dict(l=12, r=18, t=34, b=16),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=PALETTE["grid"],
    )
    fig.update_yaxes(
        gridcolor=PALETTE["grid"],
        zeroline=False,
    )
    return fig


def segment_revenue_bar(segment_summary: pd.DataFrame) -> go.Figure:
    """Bar chart of 6-month forecast revenue by segment, starting at 0 with £ and % labels."""
    ordered = segment_summary.sort_values("predicted_revenue", ascending=False).reset_index(drop=True)
    max_rev = float(ordered["predicted_revenue"].max()) if not ordered.empty else 1.0
    colors = [SEGMENT_COLORS.get(seg, PALETTE["primary"]) for seg in ordered["segment"]]
    labels = [
        f"£{rev:,.0f} ({share:.1%})"
        for rev, share in zip(ordered["predicted_revenue"], ordered["revenue_share"])
    ]

    fig = go.Figure(
        go.Bar(
            x=ordered["segment"],
            y=ordered["predicted_revenue"],
            marker=dict(color=colors, line=dict(width=0)),
            text=labels,
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack(
                [
                    ordered["customers"],
                    ordered["avg_clv"],
                    ordered["avg_churn"],
                    ordered["revenue_share"],
                ]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Forecast revenue: £%{y:,.0f} (%{customdata[3]:.1%})<br>"
                "Customers: %{customdata[0]:,}<br>"
                "Avg 6-month CLV: £%{customdata[1]:,.0f}<br>"
                "Avg inactivity risk: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(
        title_text="Six-month forecast revenue (£)",
        rangemode="tozero",
        range=[0, max(max_rev * 1.20, 1.0)],
        tickprefix="£",
        separatethousands=True,
    )
    fig.update_xaxes(title_text="")
    return style_figure(fig, height=370, show_legend=False, margin=dict(l=16, r=16, t=28, b=16))


def lorenz_concentration_chart(
    lorenz_df: pd.DataFrame,
    top_10_share: float,
    top_20_share: float,
) -> go.Figure:
    """Interactive Pareto/Lorenz concentration curve of predicted 6-month CLV."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Equal distribution",
            line=dict(color=PALETTE["muted"], dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lorenz_df["customer_pct"],
            y=lorenz_df["revenue_pct"],
            mode="lines",
            name="Forecast CLV concentration",
            line=dict(color=PALETTE["primary"], width=3),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.10)",
            customdata=np.column_stack(
                [lorenz_df["cumulative_customers"], lorenz_df["cumulative_revenue"]]
            ),
            hovertemplate=(
                "Top %{x:.1f}% of customers (%{customdata[0]:,.0f} accounts)<br>"
                "Share of forecast revenue: %{y:.1f}% (£%{customdata[1]:,.0f})"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[10.0, 20.0],
            y=[top_10_share * 100.0, top_20_share * 100.0],
            mode="markers+text",
            name="Key concentration points",
            marker=dict(color=PALETTE["accent"], size=9, line=dict(color="#ffffff", width=1.5)),
            text=[
                f"Top 10% = {top_10_share:.1%}",
                f"Top 20% = {top_20_share:.1%}",
            ],
            textposition="bottom right",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Cumulative % of customers (highest CLV first)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(title_text="Cumulative % of forecast revenue", range=[0, 104], rangemode="tozero", ticksuffix="%")
    return style_figure(fig, height=360, show_legend=True)


def segment_treemap(segment_summary: pd.DataFrame) -> go.Figure:
    """Treemap of forecast revenue split across customer segments."""
    colors = [SEGMENT_COLORS.get(seg, PALETTE["primary"]) for seg in segment_summary["segment"]]
    fig = go.Figure(
        go.Treemap(
            labels=segment_summary["segment"],
            parents=[""] * len(segment_summary),
            values=segment_summary["predicted_revenue"],
            marker=dict(colors=colors, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
            texttemplate=(
                "<b>%{label}</b><br>"
                "£%{value:,.0f} (%{percentRoot:.1%})"
            ),
            textposition="middle center",
            customdata=np.column_stack(
                [
                    segment_summary["customers"],
                    segment_summary["avg_clv"],
                    segment_summary["avg_churn"],
                ]
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Forecast revenue: £%{value:,.0f} (%{percentRoot:.1%})<br>"
                "Customers: %{customdata[0]:,}<br>"
                "Avg CLV: £%{customdata[1]:,.0f}<br>"
                "Avg churn risk: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))
    return style_figure(fig, height=360, show_legend=False, margin=dict(l=8, r=8, t=12, b=8))


def clv_churn_scatter(
    df: pd.DataFrame,
    clv_q80: float,
    clv_q50: float,
) -> go.Figure:
    """Scatter of predicted CLV (log scale) vs 90-day churn probability with segment thresholds."""
    fig = go.Figure()
    plot_df = df.copy()
    plot_df["_clv_plot"] = np.maximum(plot_df["predicted_6m_clv"].to_numpy(dtype=float), 1.0)

    present_segments = [s for s in SEGMENT_ORDER if s in plot_df["segment"].unique()]
    extra_segments = [s for s in plot_df["segment"].unique() if s not in present_segments]

    for seg in present_segments + extra_segments:
        sub = plot_df[plot_df["segment"] == seg]
        fig.add_trace(
            go.Scattergl(
                x=sub["churn_probability"] * 100.0,
                y=sub["_clv_plot"],
                mode="markers",
                name=seg,
                marker=dict(
                    color=SEGMENT_COLORS.get(seg, PALETTE["primary"]),
                    size=6,
                    opacity=0.68,
                ),
                customdata=np.column_stack(
                    [sub["Customer ID"], sub["predicted_6m_clv"], sub["churn_probability"]]
                ),
                hovertemplate=(
                    f"<b>Customer %{{customdata[0]}}</b> ({seg})<br>"
                    "Forecast 6m CLV: £%{customdata[1]:,.2f}<br>"
                    "90-day inactivity risk: %{customdata[2]:.1%}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=50.0, line_dash="dash", line_color=PALETTE["accent"], line_width=1.5)
    fig.add_vline(x=70.0, line_dash="dot", line_color=PALETTE["danger"], line_width=1.5)
    fig.add_annotation(
        x=50.0, y=1.06, yref="paper", text="50% risk", showarrow=False,
        font=dict(size=10, color=PALETTE["accent"]), xanchor="right", xshift=-4,
    )
    fig.add_annotation(
        x=70.0, y=1.06, yref="paper", text="70% Hibernating", showarrow=False,
        font=dict(size=10, color=PALETTE["danger"]), xanchor="left", xshift=4,
    )

    max_clv = float(plot_df["_clv_plot"].max()) if not plot_df.empty else 100.0
    if clv_q80 > 1.0:
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1, yref="y", y0=clv_q80, y1=clv_q80,
            line=dict(color=PALETTE["primary"], width=1.3, dash="dash"),
        )
        fig.add_annotation(
            xref="paper", x=1.02, yref="y", y=np.log10(clv_q80),
            text=f"Top 20% CLV (£{clv_q80:,.0f})",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=10, color=PALETTE["primary"]),
        )
    if clv_q50 > 1.0:
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1, yref="y", y0=clv_q50, y1=clv_q50,
            line=dict(color=PALETTE["secondary"], width=1.2, dash="dot"),
        )
        fig.add_annotation(
            xref="paper", x=1.02, yref="y", y=np.log10(clv_q50),
            text=f"Median CLV (£{clv_q50:,.0f})",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=10, color=PALETTE["secondary"]),
        )

    fig.update_xaxes(title_text="90-day inactivity risk (%)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(
        title_text="Predicted 6-month CLV (£, log scale)",
        type="log",
        range=[0, np.log10(max(max_clv * 1.8, 10.0))],
        tickprefix="£",
        separatethousands=True,
    )
    styled = style_figure(fig, height=440, show_legend=True, margin=dict(l=16, r=145, t=44, b=20))
    styled.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )
    return styled


def segment_clv_distribution(df: pd.DataFrame) -> go.Figure:
    """Violin/box distribution of positive predicted 6-month CLV by segment (log scale)."""
    fig = go.Figure()
    plot_df = df.copy()
    plot_df["_clv_plot"] = np.maximum(plot_df["predicted_6m_clv"].to_numpy(dtype=float), 1.0)
    for seg in [s for s in SEGMENT_ORDER if s in plot_df["segment"].unique()]:
        sub = plot_df[plot_df["segment"] == seg]
        fig.add_trace(
            go.Violin(
                x=[seg] * len(sub),
                y=sub["_clv_plot"],
                name=seg,
                box_visible=True,
                meanline_visible=True,
                points=False,
                line_color=SEGMENT_COLORS.get(seg, PALETTE["primary"]),
                hoverinfo="y+name",
            )
        )
    fig.update_yaxes(
        title_text="Predicted 6-month CLV (£, log scale)",
        type="log",
        tickprefix="£",
        separatethousands=True,
    )
    fig.update_xaxes(title_text="")
    return style_figure(fig, height=360, show_legend=False)


def segment_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Violin distribution of 90-day inactivity probability by segment."""
    fig = go.Figure()
    for seg in [s for s in SEGMENT_ORDER if s in df["segment"].unique()]:
        sub = df[df["segment"] == seg]
        fig.add_trace(
            go.Violin(
                x=[seg] * len(sub),
                y=sub["churn_probability"] * 100.0,
                name=seg,
                box_visible=True,
                meanline_visible=True,
                points=False,
                spanmode="hard",
                line_color=SEGMENT_COLORS.get(seg, PALETTE["primary"]),
                hoverinfo="y+name",
            )
        )
    fig.add_hline(y=50.0, line_dash="dash", line_color=PALETTE["accent"], line_width=1.4)
    fig.add_hline(y=70.0, line_dash="dot", line_color=PALETTE["danger"], line_width=1.4)
    fig.update_yaxes(
        title_text="90-day inactivity risk (%)",
        range=[0, 100],
        rangemode="tozero",
        ticksuffix="%",
    )
    fig.update_xaxes(title_text="")
    return style_figure(fig, height=360, show_legend=False)


def customer_purchase_timeline(timeline_df: pd.DataFrame) -> go.Figure:
    """Order-by-order historical spend timeline for an individual customer."""
    max_rev = float(timeline_df["Revenue"].max()) if not timeline_df.empty else 1.0
    fig = go.Figure(
        go.Bar(
            x=timeline_df["InvoiceDate"],
            y=timeline_df["Revenue"],
            marker=dict(color=PALETTE["primary"]),
            customdata=np.column_stack(
                [timeline_df["Invoice"], timeline_df["Quantity"], timeline_df["Items"]]
            ),
            hovertemplate=(
                "<b>Invoice %{customdata[0]}</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Spend: £%{y:,.2f}<br>"
                "Units: %{customdata[1]:,} · Distinct SKUs: %{customdata[2]:,}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(
        title_text="Order spend (£)",
        rangemode="tozero",
        range=[0, max(max_rev * 1.15, 1.0)],
        tickprefix="£",
        separatethousands=True,
    )
    fig.update_xaxes(title_text="Purchase date")
    return style_figure(fig, height=320, show_legend=False)


def horizontal_ranking_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    x_title: str,
    color: str = PALETTE["primary"],
    is_currency: bool = True,
    share_col: str | None = None,
    height: int = 340,
) -> go.Figure:
    """Horizontal bar chart sorted top-down with axis starting at 0."""
    ordered = df.sort_values(value_col, ascending=True).reset_index(drop=True)
    max_val = float(ordered[value_col].max()) if not ordered.empty else 1.0

    labels = []
    for _, row in ordered.iterrows():
        val = float(row[value_col])
        prefix = "£" if is_currency else ""
        base = f"{prefix}{val:,.0f}"
        if share_col and share_col in ordered.columns:
            base += f" ({float(row[share_col]):.1%})"
        labels.append(base)

    short_labels = [
        (str(lbl)[:34] + "…") if len(str(lbl)) > 35 else str(lbl)
        for lbl in ordered[label_col]
    ]

    fig = go.Figure(
        go.Bar(
            y=short_labels,
            x=ordered[value_col],
            orientation="h",
            marker=dict(color=color),
            text=labels,
            textposition="outside",
            cliponaxis=False,
            customdata=ordered[label_col],
            hovertemplate="<b>%{customdata}</b><br>Value: " + ("£" if is_currency else "") + "%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(
        title_text=x_title,
        rangemode="tozero",
        range=[0, max(max_val * 1.38, 1.0)],
        tickprefix="£" if is_currency else "",
        separatethousands=True,
        showgrid=True,
        gridcolor=PALETTE["grid"],
    )
    fig.update_yaxes(title_text="", showgrid=False, automargin=True)
    return style_figure(fig, height=height, show_legend=False, margin=dict(l=12, r=58, t=20, b=16))


def cumulative_gains_plotly(gains_df: pd.DataFrame) -> go.Figure:
    """Interactive cumulative gains curve derived from holdout decile lift."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gains_df["customer_pct"],
            y=gains_df["random_pct"],
            mode="lines",
            name="Random selection",
            line=dict(color=PALETTE["muted"], dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gains_df["customer_pct"],
            y=gains_df["revenue_pct"],
            mode="lines+markers",
            name="BG/NBD + Gamma-Gamma",
            line=dict(color=PALETTE["primary"], width=3),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.10)",
            hovertemplate=(
                "Top %{x:.0f}% of customers targeted<br>"
                "Future revenue captured: %{y:.1f}%"
                "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Share of customers targeted (%)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(title_text="Share of future revenue captured (%)", range=[0, 104], rangemode="tozero", ticksuffix="%")
    return style_figure(fig, height=360, show_legend=True)


def decile_lift_plotly(decile_df: pd.DataFrame) -> go.Figure:
    """Bar chart of holdout lift vs average across predicted CLV deciles (10 = highest)."""
    ordered = decile_df.sort_values("decile", ascending=True).reset_index(drop=True)
    max_lift = float(ordered["lift_vs_average"].max()) if not ordered.empty else 1.0
    colors = [
        PALETTE["primary"] if lift >= 1.0 else PALETTE["muted"]
        for lift in ordered["lift_vs_average"]
    ]
    text_positions = [
        "inside" if 0.55 <= float(v) < 1.0 else "outside"
        for v in ordered["lift_vs_average"]
    ]
    fig = go.Figure(
        go.Bar(
            x=[f"Decile {int(d)}" for d in ordered["decile"]],
            y=ordered["lift_vs_average"],
            marker=dict(color=colors),
            text=[f"{v:.2f}×" for v in ordered["lift_vs_average"]],
            textposition=text_positions,
            cliponaxis=False,
            customdata=np.column_stack([ordered["count"], ordered["sum"], ordered["mean"]]),
            hovertemplate=(
                "<b>%{x}</b> (10 = highest predicted CLV)<br>"
                "Lift vs average: %{y:.2f}×<br>"
                "Customers: %{customdata[0]:,}<br>"
                "Actual holdout sum: £%{customdata[1]:,.0f}<br>"
                "Actual holdout mean: £%{customdata[2]:,.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["accent"], line_width=1.4)
    fig.add_annotation(
        x=0, y=1.0, text="1.0× portfolio avg", showarrow=False,
        xanchor="left", yanchor="bottom", yshift=4, font=dict(size=10, color=PALETTE["accent"]),
    )
    fig.update_yaxes(
        title_text="Lift vs portfolio average (×)",
        rangemode="tozero",
        range=[0, max(max_lift * 1.18, 1.5)],
    )
    fig.update_xaxes(title_text="Predicted CLV decile (1 = lowest, 10 = highest)")
    return style_figure(fig, height=400, show_legend=False)


def feature_importance_plotly(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of gradient-boosting permutation feature importance."""
    ordered = importance_df.sort_values("importance", ascending=True).reset_index(drop=True)
    colors = [
        PALETTE["secondary"] if val > 0 else PALETTE["muted"]
        for val in ordered["importance"]
    ]
    fig = go.Figure(
        go.Bar(
            y=ordered["feature"],
            x=ordered["importance"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:,.1f}" for v in ordered["importance"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>MAE reduction importance: %{x:,.2f}<extra></extra>",
        )
    )
    min_val = float(ordered["importance"].min()) if not ordered.empty else 0.0
    max_val = float(ordered["importance"].max()) if not ordered.empty else 1.0
    low_bound = 0.0 if min_val >= 0 else min(min_val * 5.0, -max_val * 0.14)
    fig.update_xaxes(
        title_text="Permutation importance (mean absolute error impact, £)",
        range=[low_bound, max(max_val * 1.24, 1.0)],
        showgrid=True,
        zeroline=True,
        zerolinecolor=PALETTE["muted"],
        gridcolor=PALETTE["grid"],
    )
    fig.update_yaxes(title_text="", showgrid=False, automargin=True)
    return style_figure(fig, height=380, show_legend=False, margin=dict(l=16, r=36, t=24, b=16))


def sales_timeseries_chart(
    ts_df: pd.DataFrame,
    metric_col: str,
    y_title: str,
    is_currency: bool = True,
    show_anomalies: bool = True,
    highlight_q4: bool = True,
    color: str = PALETTE["primary"],
) -> go.Figure:
    """Time-series chart of historical sales metrics with Q4 bands and anomaly markers."""
    fig = go.Figure()
    if ts_df.empty:
        return style_figure(fig, height=330, show_legend=False)

    if highlight_q4:
        q4_years = sorted(ts_df.loc[ts_df["is_q4"], "period"].dt.year.unique())
        min_p = ts_df["period"].min()
        max_p = ts_df["period"].max()
        for yr in q4_years:
            q4_start = max(pd.Timestamp(f"{yr}-10-01"), min_p)
            q4_end = min(pd.Timestamp(f"{yr}-12-31"), max_p)
            if q4_start <= q4_end:
                fig.add_vrect(
                    x0=q4_start,
                    x1=q4_end,
                    fillcolor=PALETTE["q4_band"],
                    line=dict(color=PALETTE["q4_border"], width=1, dash="dot"),
                    annotation_text=f"Q4 {yr}",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color=PALETTE["accent"]),
                )

    prefix = "£" if is_currency else ""
    fig.add_trace(
        go.Scatter(
            x=ts_df["period"],
            y=ts_df[metric_col],
            mode="lines+markers",
            name=y_title,
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{y_title}: {prefix}%{{y:,.2f}}<extra></extra>"
            if is_currency
            else f"<b>%{{x|%Y-%m-%d}}</b><br>{y_title}: %{{y:,}}<extra></extra>",
        )
    )

    if show_anomalies and "anomaly" in ts_df.columns:
        spikes = ts_df[ts_df["anomaly"] == "High spike"]
        dips = ts_df[ts_df["anomaly"] == "Low dip"]
        if not spikes.empty:
            fig.add_trace(
                go.Scatter(
                    x=spikes["period"],
                    y=spikes[metric_col],
                    mode="markers",
                    name="Unusually high (spike)",
                    marker=dict(color=PALETTE["accent"], size=11, symbol="diamond", line=dict(color="#ffffff", width=1.2)),
                    hovertemplate=f"<b>High spike (%{{x|%Y-%m-%d}})</b><br>{y_title}: {prefix}%{{y:,.2f}}<extra></extra>",
                )
            )
        if not dips.empty:
            fig.add_trace(
                go.Scatter(
                    x=dips["period"],
                    y=dips[metric_col],
                    mode="markers",
                    name="Unusually low (dip)",
                    marker=dict(color=PALETTE["danger"], size=10, symbol="x"),
                    hovertemplate=f"<b>Low dip (%{{x|%Y-%m-%d}})</b><br>{y_title}: {prefix}%{{y:,.2f}}<extra></extra>",
                )
            )

    max_val = float(ts_df[metric_col].max()) if not ts_df.empty else 1.0
    fig.update_yaxes(
        title_text=y_title,
        rangemode="tozero",
        range=[0, max(max_val * 1.15, 1.0)],
        tickprefix=prefix,
        separatethousands=True,
    )
    fig.update_xaxes(title_text="")
    return style_figure(fig, height=340, show_legend=True)


def weekday_hour_heatmap_chart(pivot_df: pd.DataFrame) -> go.Figure:
    """Heatmap of historical order volume by weekday and hour of day."""
    fig = go.Figure(
        go.Heatmap(
            z=pivot_df.values,
            x=[f"{int(h):02d}:00" for h in pivot_df.columns],
            y=list(pivot_df.index),
            colorscale="Blues",
            hoverongaps=False,
            hovertemplate="<b>%{y} at %{x}</b><br>Orders: %{z:,}<extra></extra>",
            colorbar=dict(title="Orders"),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Hour of day", showgrid=False)
    return style_figure(fig, height=340, show_legend=False, margin=dict(l=16, r=16, t=24, b=16))
