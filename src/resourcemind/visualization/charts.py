"""Plotly figures. Pure functions of computed results (no computation of their own).

Status is always encoded redundantly: colour + marker shape + text label.
The map is a schematic km grid drawn offline (no map tiles, no network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from resourcemind.allocation.models import AllocationProblem
from resourcemind.data.generate import MAP_HEIGHT_KM, MAP_WIDTH_KM, RIVER_X_KM

COLORS = {
    "critical": "#EF4444",
    "strained": "#F59E0B",
    "covered": "#22C55E",
    "neutral": "#94A3B8",
    "accent": "#F59E0B",
    "history": "#CBD5E1",
    "forecast": "#F59E0B",
    "band": "rgba(245, 158, 11, 0.18)",
    "grid": "rgba(148, 163, 184, 0.15)",
    "river": "rgba(56, 189, 248, 0.35)",
    "move": "#38BDF8",
    "blocked": "#EF4444",
}
STRATEGY_COLORS = {"current": "#64748B", "greedy": "#38BDF8", "optimized": "#F59E0B"}
STATUS_STYLE = {
    # status: (colour, marker symbol, label)
    "critical": (COLORS["critical"], "triangle-up", "Critical (<80% covered)"),
    "strained": (COLORS["strained"], "diamond", "Strained (80-99% covered)"),
    "covered": (COLORS["covered"], "circle", "Covered (100%)"),
}
FONT = "Inter, Segoe UI, system-ui, sans-serif"


def coverage_status(coverage: float) -> str:
    if coverage >= 0.999:
        return "covered"
    return "strained" if coverage >= 0.8 else "critical"


def _style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title={"text": title, "font": {"size": 15}} if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": "#E5E7EB", "size": 13},
        margin={"l": 10, "r": 10, "t": 40 if title else 10, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"font": {"family": FONT}},
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


def demand_trend(
    history_wide: pd.DataFrame, forecast_daily: pd.DataFrame, zone: str | None = None, days: int = 90
) -> go.Figure:
    """Observed history (scenario-adjusted) and forecast with the empirical interval."""
    if zone:
        hist = history_wide[zone]
        fc = forecast_daily[forecast_daily["zone_id"] == zone].sort_values("target_date")
        fc = fc.set_index("target_date")[["predicted", "lower", "upper"]]
    else:
        hist = history_wide.sum(axis=1)
        fc = forecast_daily.groupby("target_date")[["predicted", "lower", "upper"]].sum()
    hist = hist.iloc[-days:]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            name="Observed (synthetic)",
            mode="lines",
            line={"color": COLORS["history"], "width": 1.6},
            hovertemplate="%{x|%a %d %b}: %{y:.0f} requests<extra></extra>",
        )
    )
    hist_rolling = hist.rolling(7, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist_rolling.values,
            name="7-day mean",
            mode="lines",
            line={"color": COLORS["neutral"], "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc["upper"]) + list(fc["lower"][::-1]),
            fill="toself",
            mode="lines",
            fillcolor=COLORS["band"],
            line={"width": 0},
            name="Empirical interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=fc["predicted"],
            name="Forecast",
            mode="lines+markers",
            line={"color": COLORS["forecast"], "width": 2.4, "dash": "dash"},
            marker={"symbol": "diamond", "size": 7},
            hovertemplate="%{x|%a %d %b}: %{y:.1f} forecast<extra></extra>",
        )
    )
    fig.update_yaxes(title="Service requests / day", rangemode="tozero")
    return _style(fig, 340)


def zone_map(
    zones: pd.DataFrame,
    problem: AllocationProblem,
    x: np.ndarray,
    served: np.ndarray,
    show_moves: bool = True,
) -> go.Figure:
    """Schematic map: marker size = demand, colour/shape = coverage status, arrows = moves."""
    z = zones.set_index("zone_id").loc[list(problem.zone_ids)].reset_index()
    stationed = x.sum(axis=0)
    coverage = np.where(problem.demand > 0, served / np.maximum(problem.demand, 1), 1.0)
    statuses = [coverage_status(c) for c in coverage]
    fig = go.Figure()

    # River (schematic) and blocked routes
    fig.add_shape(
        type="line",
        x0=RIVER_X_KM,
        x1=RIVER_X_KM,
        y0=0,
        y1=MAP_HEIGHT_KM,
        line={"color": COLORS["river"], "width": 10},
        layer="below",
    )
    fig.add_annotation(x=RIVER_X_KM, y=MAP_HEIGHT_KM - 0.6, text="river", showarrow=False, font={"color": "#7DD3FC"})
    xy = z[["x_km", "y_km"]].to_numpy()
    n = problem.n
    blocked_drawn = False
    for j in range(n):
        for i in range(j + 1, n):
            explicitly_blocked = not problem.eligible[j, i] and problem.travel[j, i] <= problem.max_travel_minutes
            if explicitly_blocked:
                fig.add_trace(
                    go.Scatter(
                        x=[xy[j, 0], xy[i, 0]],
                        y=[xy[j, 1], xy[i, 1]],
                        mode="lines",
                        line={"color": COLORS["blocked"], "width": 1, "dash": "dot"},
                        name="Blocked route / inaccessible",
                        legendgroup="blocked",
                        showlegend=not blocked_drawn,
                        hoverinfo="skip",
                        opacity=0.5,
                    )
                )
                blocked_drawn = True

    if show_moves:
        for j in range(n):
            for i in range(n):
                if i != j and x[j, i] > 0:
                    fig.add_annotation(
                        x=xy[i, 0],
                        y=xy[i, 1],
                        ax=xy[j, 0],
                        ay=xy[j, 1],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.4,
                        arrowwidth=1.5 + x[j, i],
                        arrowcolor=COLORS["move"],
                        standoff=14,
                        startstandoff=10,
                        text="",
                    )
        if (x * (1 - np.eye(n, dtype=int))).sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line={"color": COLORS["move"], "width": 3},
                    name="Team movement (arrow)",
                )
            )

    size = 18 + 42 * np.sqrt(problem.demand / max(problem.demand.max(), 1))
    for status, (color, symbol, label) in STATUS_STYLE.items():
        idx = [k for k, s in enumerate(statuses) if s == status]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter(
                x=xy[idx, 0],
                y=xy[idx, 1],
                mode="markers+text",
                name=label,
                marker={
                    "size": size[idx],
                    "color": color,
                    "symbol": symbol,
                    "opacity": 0.85,
                    "line": {"color": "#0B0D10", "width": 1.5},
                },
                text=[f"{problem.zone_ids[k]} · {stationed[k]}T · {coverage[k]:.0%}" for k in idx],
                textposition="bottom center",
                textfont={"size": 12, "color": "#E5E7EB"},
                customdata=np.stack(
                    [
                        [problem.zone_names[k] for k in idx],
                        problem.demand[idx],
                        stationed[idx],
                        np.round(coverage[idx] * 100, 1),
                        problem.weights[idx],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Demand: %{customdata[1]} requests<br>"
                    "Teams: %{customdata[2]}<br>Coverage: %{customdata[3]}%<br>"
                    "Priority weight: %{customdata[4]}<extra></extra>"
                ),
            )
        )
    inacc = [k for k in range(n) if problem.eligible[:, k].sum() == 1 and problem.eligible[k, :].sum() == 1]
    if inacc:
        fig.add_trace(
            go.Scatter(
                x=xy[inacc, 0],
                y=xy[inacc, 1],
                mode="markers",
                name="Isolated / inaccessible (X)",
                marker={"symbol": "x-thin", "size": 30, "line": {"color": "#F8FAFC", "width": 3}},
                hoverinfo="skip",
            )
        )
    # Fixed ranges keep the schematic stable; "domain" constraint keeps 1 km = 1 km
    # without letting Plotly widen the range on narrow screens.
    fig.update_xaxes(range=[0, MAP_WIDTH_KM], title="km (schematic, synthetic)", constrain="domain", fixedrange=True)
    fig.update_yaxes(
        range=[0, MAP_HEIGHT_KM], title="km", scaleanchor="x", scaleratio=1, constrain="domain", fixedrange=True
    )
    return _style(fig, 520)


def allocation_comparison(problem: AllocationProblem, x_before: np.ndarray, x_after: np.ndarray) -> go.Figure:
    labels = [f"{z}<br>{nm}" for z, nm in zip(problem.zone_ids, problem.zone_names, strict=True)]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=x_before.sum(axis=0),
        name="Current teams",
        marker_color=STRATEGY_COLORS["current"],
        text=x_before.sum(axis=0),
        textposition="outside",
        cliponaxis=False,
    )
    fig.add_bar(
        x=labels,
        y=x_after.sum(axis=0),
        name="Recommended teams",
        marker_color=STRATEGY_COLORS["optimized"],
        marker_pattern_shape="/",
        text=x_after.sum(axis=0),
        textposition="outside",
        cliponaxis=False,
    )
    fig.add_scatter(
        x=labels,
        y=problem.staging_capacity,
        mode="markers",
        name="Staging capacity (max)",
        marker={"symbol": "line-ew-open", "size": 26, "color": "#E5E7EB", "line": {"width": 2}},
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Response teams")
    return _style(fig, 380)


def strategy_bars(comparison: pd.DataFrame, metric: str, title: str, fmt: str = ".1f") -> go.Figure:
    df = comparison.dropna(subset=[metric]) if metric in comparison else comparison.iloc[0:0]
    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=df[metric],
            marker_color=[STRATEGY_COLORS[s] for s in df["strategy"]],
            text=[format(v, fmt) for v in df[metric]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, 300, title)


def zone_coverage_chart(zone_tables: dict[str, pd.DataFrame], labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for key, table in zone_tables.items():
        fig.add_bar(
            x=table["zone_id"],
            y=table["coverage"] * 100,
            name=labels.get(key, key),
            marker_color=STRATEGY_COLORS.get(key, COLORS["neutral"]),
            hovertemplate="%{x}: %{y:.1f}%<extra>" + labels.get(key, key) + "</extra>",
        )
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["critical"], annotation_text="80%")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Coverage (%)", range=[0, 105])
    return _style(fig, 340)


def model_metric_chart(metrics: pd.DataFrame, metric: str = "mae", title: str = "") -> go.Figure:
    df = metrics.sort_values(metric)
    fig = go.Figure(
        go.Bar(
            x=df[metric],
            y=df["model"],
            orientation="h",
            marker_color=[COLORS["forecast"] if m == "gradient_boosting" else COLORS["neutral"] for m in df["model"]],
            text=[f"{v:.2f}" for v in df[metric]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(title=metric.upper(), rangemode="tozero")
    return _style(fig, 260, title)


def horizon_error_chart(per_horizon: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    symbols = ["circle", "square", "diamond", "triangle-up"]
    for k, (model, grp) in enumerate(per_horizon.groupby("model")):
        fig.add_scatter(
            x=grp["horizon"],
            y=grp["mae"],
            mode="lines+markers",
            name=model,
            marker={"symbol": symbols[k % len(symbols)], "size": 8},
        )
    fig.update_xaxes(title="Horizon (days ahead)", dtick=1)
    fig.update_yaxes(title="Test MAE (requests/day)", rangemode="tozero")
    return _style(fig, 300)


def backtest_chart(test_predictions: pd.DataFrame, zone: str, models: list[str], horizon: int = 1) -> go.Figure:
    df = test_predictions[(test_predictions["zone_id"] == zone) & (test_predictions["horizon"] == horizon)]
    df = df.sort_values("target_date")
    fig = go.Figure()
    fig.add_scatter(
        x=df["target_date"],
        y=df["y"],
        name="Actual (held-out test)",
        mode="lines+markers",
        line={"color": COLORS["history"]},
    )
    dashes = ["dash", "dot", "dashdot", "longdash"]
    for k, m in enumerate(models):
        fig.add_scatter(x=df["target_date"], y=df[m], name=m, mode="lines", line={"dash": dashes[k % 4]})
    fig.update_yaxes(title="Requests / day", rangemode="tozero")
    return _style(fig, 320)


def sensitivity_chart(sweep: pd.DataFrame, metric: str, x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    symbols = {"current": "square", "greedy": "diamond", "optimized": "circle"}
    for key, grp in sweep.groupby("strategy", sort=False):
        grp = grp.sort_values("factor")
        fig.add_scatter(
            x=grp["factor"],
            y=grp[metric],
            mode="lines+markers",
            name=grp["label"].iloc[0],
            line={"color": STRATEGY_COLORS[key]},
            marker={"symbol": symbols[key], "size": 8},
        )
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label, rangemode="tozero")
    return _style(fig, 320)
