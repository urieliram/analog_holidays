"""Plot helpers for the holiday audit Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import HOUR_COLS

HOURS = list(range(24))

LABEL_COLOR = {
    "holiday": "#2ca02c",
    "special_day": "#d62728",
    "normal_day": "#1f77b4",
}

LABEL_SYMBOL = {
    "holiday": "🟢",
    "special_day": "🔴",
    "normal_day": "🔵",
}

DOW_SHORT_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_HOVER_TMPL = "Hour %{x}<br>%{y:,.1f} MW<extra></extra>"


def plot_day_hourly(
    row: pd.Series,
    title: str = "",
    df_context: pd.DataFrame | None = None,
    df_group: pd.DataFrame | None = None,
) -> go.Figure:
    """Plot one daily curve, or a short multi-day window when context is available."""
    label = str(row.get("label", "normal_day"))
    color = LABEL_COLOR.get(label, "#888")
    h_name = row.get("holiday_name") or ""

    if df_context is not None and not df_context.empty:
        selected_date = pd.Timestamp(row.get("date", pd.NaT)).normalize()
        fig = go.Figure()

        for _, context_row in df_context.sort_values("date").iterrows():
            current_date = pd.Timestamp(context_row["date"]).normalize()
            current_label = str(context_row.get("label", "normal_day"))
            current_color = LABEL_COLOR.get(current_label, "#888")
            is_selected = current_date == selected_date

            x_values = [current_date + pd.Timedelta(hours=h) for h in range(24)]
            y_values = [float(context_row.get(h, np.nan)) for h in HOUR_COLS]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines" + ("+markers" if is_selected else ""),
                    line=dict(color=current_color, width=3 if is_selected else 1.2),
                    marker=dict(size=4, color=current_color) if is_selected else dict(size=0),
                    opacity=1.0 if is_selected else 0.55,
                    showlegend=False,
                    hovertemplate=(
                        f"{current_date.date()} %{{x|%H:%M}}<br>%{{y:,.1f}} MW<extra></extra>"
                    ),
                )
            )

        fig.add_vrect(
            x0=selected_date,
            x1=selected_date + pd.Timedelta(hours=23, minutes=59),
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="#333333", width=2),
        )

        all_days = sorted(df_context["date"].apply(pd.Timestamp).unique())
        for day_value in all_days[1:]:
            fig.add_vline(
                x=day_value.timestamp() * 1000,
                line=dict(color="#cccccc", width=1, dash="dot"),
            )

        tick_vals = [pd.Timestamp(day_value).normalize() + pd.Timedelta(hours=12) for day_value in all_days]
        tick_text = [pd.Timestamp(day_value).strftime("%a\n%d %b") for day_value in all_days]

        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(size=10),
                showgrid=False,
            ),
            yaxis=dict(title="Demand (MW)"),
            height=360,
            margin=dict(l=55, r=20, t=55, b=45),
            showlegend=False,
            plot_bgcolor="#fafafa",
        )
        return fig

    y_vals = [row.get(h, np.nan) for h in HOUR_COLS]
    fig = go.Figure()

    if df_group is not None and not df_group.empty:
        profiles: list[list[float]] = []
        current_date_ts = pd.Timestamp(row.get("date", pd.NaT)).normalize()
        for _, group_row in df_group.iterrows():
            other_date = pd.Timestamp(group_row["date"]).normalize()
            if other_date == current_date_ts:
                continue
            y_group = [float(group_row.get(h, np.nan)) for h in HOUR_COLS]
            other_color = LABEL_COLOR.get(str(group_row.get("label", "normal_day")), "#888")
            fig.add_trace(
                go.Scatter(
                    x=HOURS,
                    y=y_group,
                    mode="lines",
                    line=dict(color=other_color, width=0.8),
                    opacity=0.18,
                    showlegend=False,
                    hovertemplate=f"{other_date.date()}<br>" + _HOVER_TMPL,
                )
            )
            if not any(np.isnan(value) for value in y_group):
                profiles.append(y_group)
        if profiles:
            centroid = np.nanmean(profiles, axis=0).tolist()
            fig.add_trace(
                go.Scatter(
                    x=HOURS,
                    y=centroid,
                    mode="lines",
                    line=dict(color="black", width=1.8, dash="dot"),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate="Centroid<br>" + _HOVER_TMPL,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=HOURS,
            y=y_vals,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            name=label,
            hovertemplate=_HOVER_TMPL,
        )
    )
    subtitle = f"{h_name} · {label.replace('_', ' ')}" if h_name else label.replace("_", " ")
    fig.update_layout(
        title=dict(text=title or subtitle, font=dict(size=15)),
        xaxis=dict(
            title="Hour",
            tickvals=list(range(0, 24, 3)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
        ),
        yaxis=dict(title="Demand (MW)"),
        height=330,
        margin=dict(l=55, r=20, t=55, b=45),
        showlegend=False,
        plot_bgcolor="#fafafa",
    )
    return fig


def plot_weekly_context(
    df: pd.DataFrame,
    region: str,
    current_date: pd.Timestamp,
    n_days: int = 7,
) -> go.Figure:
    """Plot a compact row of nearby days centered on the current one."""
    region_df = df[df["unique_id"] == region].sort_values("date")
    dates = list(region_df["date"].unique())
    if not dates:
        return go.Figure()

    try:
        current_index = next(
            i for i, day_value in enumerate(dates)
            if pd.Timestamp(day_value) == pd.Timestamp(current_date)
        )
    except StopIteration:
        current_index = 0

    half = n_days // 2
    start = max(0, current_index - half)
    end = min(len(dates), start + n_days)
    start = max(0, end - n_days)
    window = dates[start:end]

    subplot_titles = []
    for day_value in window:
        day_row = region_df[region_df["date"] == day_value].iloc[0]
        dow = DOW_SHORT_EN[int(day_row["dow"])]
        star = " ★" if pd.Timestamp(day_value) == pd.Timestamp(current_date) else ""
        subplot_titles.append(f"{dow}<br>{pd.Timestamp(day_value).strftime('%m/%d')}{star}")

    fig = make_subplots(
        rows=1,
        cols=len(window),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
    )

    for index, day_value in enumerate(window):
        day_rows = region_df[region_df["date"] == day_value]
        if day_rows.empty:
            continue
        day_row = day_rows.iloc[0]
        y_vals = [day_row.get(h, np.nan) for h in HOUR_COLS]
        label = str(day_row.get("label", "normal_day"))
        color = LABEL_COLOR.get(label, "#888")
        is_current = pd.Timestamp(day_value) == pd.Timestamp(current_date)
        line_width = 2.5 if is_current else 1.2
        opacity = 1.0 if is_current else 0.6

        fig.add_trace(
            go.Scatter(
                x=HOURS,
                y=y_vals,
                mode="lines",
                line=dict(color=color, width=line_width),
                opacity=opacity,
                showlegend=False,
                hovertemplate=f"{pd.Timestamp(day_value).date()}<br>" + _HOVER_TMPL,
            ),
            row=1,
            col=index + 1,
        )
        if is_current:
            fig.add_vrect(
                x0=-0.5, x1=23.5,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="#333333", width=2),
                row=1,
                col=index + 1,
            )

    fig.update_layout(
        height=210,
        margin=dict(l=5, r=5, t=45, b=10),
        plot_bgcolor="#fafafa",
    )
    fig.update_xaxes(tickvals=[0, 12, 23], tickfont=dict(size=8), showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor="#eeeeee")
    return fig


def plot_cluster_reference(
    df: pd.DataFrame,
    region: str,
    current_month: int,
    current_dow: int,
    selected_date: pd.Timestamp | None = None,
) -> go.Figure:
    """Plot historical weekday profiles for the current month in one row."""
    region_df = df[(df["unique_id"] == region) & (df["month"] == current_month)]

    month_name = MONTH_ABBR[current_month - 1]
    titles = [DOW_SHORT_EN[dow] for dow in range(7)]

    fig = make_subplots(
        rows=1,
        cols=7,
        subplot_titles=titles,
        horizontal_spacing=0.018,
        shared_yaxes=True,
    )

    for column_index, dow in enumerate(range(7)):
        dow_df = region_df[region_df["dow"] == dow].copy()
        is_current_dow = dow == current_dow
        profiles: list[list[float]] = []

        for _, day_row in dow_df.iterrows():
            y_vals = [float(day_row.get(h, np.nan)) for h in HOUR_COLS]
            label = str(day_row.get("label", "normal_day"))
            color = LABEL_COLOR.get(label, "#888")

            is_today = (
                selected_date is not None
                and pd.Timestamp(day_row["date"]).normalize() == pd.Timestamp(selected_date).normalize()
            )

            line_width = 2.5 if is_today else 0.9
            opacity = 1.0 if is_today else 0.25

            fig.add_trace(
                go.Scatter(
                    x=HOURS,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    opacity=opacity,
                    showlegend=False,
                    hovertemplate=f"{pd.Timestamp(day_row['date']).date()}<br>" + _HOVER_TMPL,
                ),
                row=1,
                col=column_index + 1,
            )
            if not any(np.isnan(value) for value in y_vals):
                profiles.append(y_vals)

        if profiles:
            centroid = np.nanmean(profiles, axis=0).tolist()
            fig.add_trace(
                go.Scatter(
                    x=HOURS, y=centroid,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    opacity=0.85,
                    showlegend=False,
                    hovertemplate=f"Centroid · {month_name} {DOW_SHORT_EN[dow]}<br>" + _HOVER_TMPL,
                ),
                row=1,
                col=column_index + 1,
            )

        if is_current_dow:
            fig.add_vrect(
                x0=-0.5, x1=23.5,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="#333333", width=2),
                row=1,
                col=column_index + 1,
            )

    fig.update_layout(
        title=dict(
            text=f"{month_name} — historical profiles by weekday  "
                 f"<span style='color:#888;font-size:11px'>"
                 f"(🟢 holiday · 🔴 special_day · 🔵 normal_day · — centroid)</span>",
            font=dict(size=13),
        ),
        height=255,
        margin=dict(l=30, r=10, t=52, b=8),
        plot_bgcolor="#fafafa",
    )
    fig.update_xaxes(
        tickvals=[0, 6, 12, 18, 23],
        ticktext=["0h", "6h", "12h", "18h", "23h"],
        tickfont=dict(size=7),
        showgrid=False,
    )
    fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor="#eeeeee")
    return fig


def build_reference_table(
    df: pd.DataFrame,
    region: str,
    month: int,
    dow: int,
) -> pd.DataFrame:
    """Return mean hourly profiles by label for one `(region, month, dow)` group."""
    mask = (df["unique_id"] == region) & (df["month"] == month) & (df["dow"] == dow)
    subset = df[mask]

    rows = []
    for label in ("holiday", "special_day", "normal_day"):
        label_rows = subset[subset["label"] == label]
        values = (
            [label_rows[h].mean() for h in HOUR_COLS]
            if not label_rows.empty
            else [np.nan] * 24
        )
        row = {"type": label}
        for index, hour_col in enumerate(HOUR_COLS):
            row[hour_col] = round(values[index], 1) if not np.isnan(values[index]) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("type")
