"""ResourceMind AI - Streamlit dashboard.

Run:  streamlit run app.py

Every number on screen comes from resourcemind.pipeline.run_pipeline: real
forecasting models, a real CP-SAT optimiser and an independent validator.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from resourcemind.allocation.models import compute_outcome, movements_table  # noqa: E402
from resourcemind.config import AppConfig, config_from_dict, load_config  # noqa: E402
from resourcemind.data.generate import generate_dataset  # noqa: E402
from resourcemind.data.validate import DataValidationError  # noqa: E402
from resourcemind.forecasting.evaluation import evaluate_forecasters  # noqa: E402
from resourcemind.forecasting.models import MODEL_REGISTRY, model_label  # noqa: E402
from resourcemind.metrics.evaluation import tier_table  # noqa: E402
from resourcemind.pipeline import (  # noqa: E402
    STRATEGY_LABELS,
    PipelineResult,
    PipelineSettings,
    run_pipeline,
    run_record,
    run_summary_markdown,
    zone_plan_table,
)
from resourcemind.simulation.scenarios import PRESETS, ScenarioOverrides  # noqa: E402
from resourcemind.simulation.sensitivity import PARAMETERS, sensitivity_sweep  # noqa: E402
from resourcemind.ui import components as ui  # noqa: E402
from resourcemind.visualization import charts  # noqa: E402

st.set_page_config(
    page_title="ResourceMind AI",
    page_icon=":material/emergency:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

BASE_CONFIG = load_config()
PAGES = [
    "Command overview",
    "Demand forecasts",
    "Allocation planner",
    "Scenario lab",
    "Strategy evaluation",
    "Methodology",
]
CONSTRAINTS = [
    "resource conservation",
    "minimum retention",
    "route eligibility",
    "staging capacity",
    "coverage floor",
    "integrality",
    "non-negativity",
]


# --------------------------------------------------------------------------- cached computation
@st.cache_data(show_spinner=False, max_entries=6)
def get_dataset(seed: int):
    return generate_dataset(BASE_CONFIG.with_updates(seed=seed))


@st.cache_data(show_spinner=False, max_entries=12)
def get_forecast_eval(seed: int, horizon: int):
    cfg = BASE_CONFIG.with_updates(seed=seed, horizon_days=horizon)
    return evaluate_forecasters(get_dataset(seed).demand_wide(), horizon, cfg.forecast, random_state=seed)


@st.cache_data(show_spinner=False, max_entries=48)
def get_result(config_json: str, scenario: str, model: str, overrides_json: str, replace: bool) -> PipelineResult:
    cfg = config_from_dict(json.loads(config_json))
    settings = PipelineSettings(
        scenario=scenario,
        model=model,
        overrides=ScenarioOverrides.from_dict(json.loads(overrides_json)),
        replace_preset=replace,
    )
    return run_pipeline(
        cfg, settings, dataset=get_dataset(cfg.seed), forecast_eval=get_forecast_eval(cfg.seed, cfg.horizon_days)
    )


@st.cache_data(show_spinner=False, max_entries=24)
def get_sensitivity(
    config_json: str, scenario: str, model: str, overrides_json: str, replace: bool, parameter: str, factors: tuple
):
    result = get_result(config_json, scenario, model, overrides_json, replace)
    return sensitivity_sweep(result.problem, config_from_dict(json.loads(config_json)), parameter, list(factors))


# --------------------------------------------------------------------------- state helpers
def lab_state() -> dict | None:
    """Scenario-lab overrides (complete spec) or None when the preset is used unchanged."""
    return st.session_state.get("lab_overrides")


def reset_lab() -> None:
    st.session_state["lab_overrides"] = None
    for key in [k for k in st.session_state if str(k).startswith("lab_w_")]:
        del st.session_state[key]
    st.session_state.pop("sensitivity", None)


def fmt_int(v) -> str:
    return f"{v:,.0f}"


def pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


# --------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[str, AppConfig, str, str]:
    with st.sidebar:
        st.markdown("## ResourceMind AI")
        st.caption("Adaptive emergency resource allocation · decision-support prototype · synthetic data")
        page = st.radio("Section", PAGES, key="page")
        st.divider()
        scenario = st.selectbox(
            "Scenario",
            list(PRESETS),
            format_func=lambda k: PRESETS[k].name,
            key="scenario",
            on_change=reset_lab,
            help="Each preset changes the actual inputs: history, resources or travel network.",
        )
        with st.expander("Planning settings"):
            seed = st.number_input("Random seed (synthetic data)", 0, 999_999, BASE_CONFIG.seed, key="seed")
            horizon = st.slider("Forecast & planning horizon (days)", 1, 7, BASE_CONFIG.horizon_days, key="horizon")
            model = st.selectbox(
                "Forecast model",
                ["auto", *MODEL_REGISTRY],
                format_func=lambda m: "Auto (best validation MAE)" if m == "auto" else model_label(m),
                key="model",
            )
            basis = st.radio(
                "Plan against",
                ["point", "upper"],
                format_func=lambda b: "Point forecast" if b == "point" else "Upper empirical interval (cautious)",
                key="basis",
            )
            mode = st.radio(
                "Optimizer objective",
                ["lexicographic", "weighted"],
                format_func=lambda m: (
                    "Lexicographic: coverage first, then travel"
                    if m == "lexicographic"
                    else "Weighted sum: coverage + λ·travel"
                ),
                key="mode",
            )
            lam = st.number_input(
                "λ: weighted requests traded per team-minute",
                min_value=0.0,
                max_value=5.0,
                value=BASE_CONFIG.optimizer.travel_penalty_lambda,
                step=0.005,
                format="%.3f",
                disabled=mode != "weighted",
                key="lam",
            )
            time_limit = st.slider("Solver time limit (s)", 1, 30, int(BASE_CONFIG.optimizer.time_limit_seconds))
        if lab_state() is not None:
            st.warning("Scenario-lab overrides are active.")
            st.button("Reset to preset defaults", on_click=reset_lab, width="stretch")
    optimizer = BASE_CONFIG.optimizer.__class__(
        mode=mode,
        travel_penalty_lambda=float(lam),
        time_limit_seconds=float(time_limit),
        num_workers=BASE_CONFIG.optimizer.num_workers,
    )
    cfg = BASE_CONFIG.with_updates(seed=int(seed), horizon_days=int(horizon), planning_basis=basis, optimizer=optimizer)
    return page, cfg, scenario, model


def export_panel(r: PipelineResult) -> None:
    with st.sidebar, st.expander("Export results"):
        st.download_button(
            "Zone forecasts (CSV)",
            r.forecast.daily.merge(r.demand, on="zone_id").to_csv(index=False),
            "resourcemind_forecasts.csv",
            "text/csv",
            width="stretch",
        )
        if r.runs["optimized"].outcome is not None:
            st.download_button(
                "Allocation plan (CSV)",
                zone_plan_table(r).to_csv(index=False),
                "resourcemind_allocation.csv",
                "text/csv",
                width="stretch",
            )
            st.download_button(
                "Team movements (CSV)",
                movements_table(r.problem, r.runs["optimized"].result.x).to_csv(index=False),
                "resourcemind_movements.csv",
                "text/csv",
                width="stretch",
            )
        st.download_button(
            "Strategy comparison (CSV)",
            r.comparison.to_csv(index=False),
            "resourcemind_strategies.csv",
            "text/csv",
            width="stretch",
        )
        st.download_button(
            "Run record (JSON)",
            json.dumps(run_record(r), indent=2, default=str),
            "resourcemind_run.json",
            "application/json",
            width="stretch",
        )
        st.download_button(
            "Run summary (Markdown)",
            run_summary_markdown(r),
            "resourcemind_summary.md",
            "text/markdown",
            width="stretch",
        )


# --------------------------------------------------------------------------- shared blocks
def status_strip(r: PipelineResult) -> None:
    opt = r.runs["optimized"]
    validated = opt.outcome is not None and not opt.violations
    parts = [
        ui.pill(f"Solver: {opt.result.status}", ui.status_tone(opt.result.status)),
        ui.pill(
            "Constraints verified independently" if validated else "No validated plan",
            "good" if validated else "critical",
        ),
        ui.pill(f"Forecast: {r.forecast.model}", "info"),
        ui.pill("Synthetic data", "warning"),
    ]
    st.markdown(" ".join(parts), unsafe_allow_html=True)


def plan_or_current(r: PipelineResult):
    """Recommended plan when one exists, otherwise the current allocation (clearly labelled)."""
    opt = r.runs["optimized"]
    if opt.outcome is not None and not opt.violations:
        return opt.result.x, opt.outcome, True
    cur = r.runs["current"]
    return cur.result.x, cur.outcome, False


def infeasible_box(r: PipelineResult) -> None:
    opt = r.runs["optimized"]
    st.error(
        f"The optimizer did not produce a valid plan (status **{opt.result.status}**). "
        "No recommendation is shown as feasible."
    )
    for line in opt.result.messages + opt.result.diagnostics.get("infeasibility_reasons", []):
        st.markdown(f"- {line}")


def map_table(r: PipelineResult, x: np.ndarray, label: str) -> pd.DataFrame:
    p = r.problem
    o = compute_outcome(p, x)
    cov = np.where(p.demand > 0, o.served / np.maximum(p.demand, 1), 1.0)
    return pd.DataFrame(
        {
            "Zone": [f"{z} {n}" for z, n in zip(p.zone_ids, p.zone_names, strict=True)],
            "Status": [charts.STATUS_STYLE[charts.coverage_status(c)][2] for c in cov],
            "Demand (requests)": p.demand,
            "Current teams": p.available,
            f"{label} teams": o.stationed,
            "Coverage": cov,
            "Priority weight": p.weights,
            "Accessible": ["No" if v else "Yes" for v in r.gaps["inaccessible"]],
        }
    )


# --------------------------------------------------------------------------- pages
def page_overview(r: PipelineResult) -> None:
    st.title("Command overview")
    ui.disclaimer()
    st.markdown(f"**{r.inputs.name}.** {r.inputs.description}")
    if r.inputs.changes:
        st.caption("Inputs changed from defaults: " + "; ".join(r.inputs.changes))
    status_strip(r)

    p, cmp = r.problem, r.comparison.set_index("strategy")
    cur = cmp.loc["current"]
    opt_ok = r.runs["optimized"].outcome is not None and not r.runs["optimized"].violations
    opt = cmp.loc["optimized"] if opt_ok else None
    H = r.config.horizon_days
    cov_opt = opt["coverage"] if opt is not None else None
    cards = [
        {
            "label": f"Planning demand · next {H} d",
            "value": fmt_int(p.demand.sum()),
            "sub": f"requests · interval {r.demand['lower_total'].sum():,.0f}-{r.demand['upper_total'].sum():,.0f}",
            "tone": "info",
        },
        {
            "label": "Teams available",
            "value": fmt_int(p.available.sum()),
            "sub": f"capacity {p.available.sum() * p.service_per_team:,} requests",
            "tone": "warning" if p.available.sum() * p.service_per_team < p.servable.sum() else "neutral",
        },
        {
            "label": "Expected coverage (plan)",
            "value": pct(cov_opt),
            "sub": f"current allocation {pct(cur['coverage'])}",
            "tone": "neutral"
            if cov_opt is None
            else ("good" if cov_opt >= 0.95 else "warning" if cov_opt >= 0.8 else "critical"),
        },
        {
            "label": "Weighted unmet demand",
            "value": f"{opt['weighted_unmet']:,.1f}" if opt is not None else "n/a",
            "sub": f"current {cur['weighted_unmet']:,.1f}",
            "tone": "neutral"
            if opt is None
            else ("good" if opt["weighted_unmet"] < cur["weighted_unmet"] else "neutral"),
        },
        {
            "label": "Teams to reposition",
            "value": fmt_int(opt["teams_moved"]) if opt is not None else "n/a",
            "sub": f"{opt['travel_minutes']:,.0f} team-minutes travel" if opt is not None else "no valid plan",
            "tone": "info",
        },
    ]
    ui.cards(cards)
    if not opt_ok:
        infeasible_box(r)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("System demand trend")
        st.plotly_chart(charts.demand_trend(r.inputs.demand_wide(), r.forecast.daily), width="stretch")
        st.caption(
            f"Last 90 days of scenario history and the {H}-day forecast ({r.forecast.model}). "
            "Shaded band: empirical interval from validation residuals."
        )
    with right:
        st.subheader("Plan summary")
        for line in r.summary_lines:
            st.markdown(f"- {line}")
        short = r.gaps[r.gaps["gap"] > 0].sort_values("gap", ascending=False)
        st.markdown("**Largest pre-allocation shortages**")
        if short.empty:
            st.success("No zone has forecast demand above its current team capacity.")
        else:
            st.dataframe(
                short[["zone_id", "zone_name", "planning_demand", "current_capacity", "gap"]].head(5),
                hide_index=True,
                width="stretch",
                column_config={
                    "planning_demand": "Demand",
                    "current_capacity": "Capacity",
                    "gap": st.column_config.NumberColumn("Shortage", help="requests over the horizon"),
                },
            )

    st.subheader("Shortage map (schematic)")
    x, outcome, is_plan = plan_or_current(r)
    view = "Recommended plan"
    if is_plan:
        view = st.segmented_control(
            "Map shows", ["Recommended plan", "Current allocation"], default="Recommended plan", key="map_view"
        )
    if view == "Current allocation" or not is_plan:
        x, outcome = r.runs["current"].result.x, r.runs["current"].outcome
    st.plotly_chart(charts.zone_map(r.inputs.zones, r.problem, x, outcome.served, show_moves=is_plan), width="stretch")
    st.caption(
        "Marker size = forecast demand; colour and shape = coverage status; label = zone · teams · coverage. "
        "Arrows = recommended team movements. Synthetic coordinates, not a real place."
    )
    with st.expander("Accessible table version of the map", expanded=False):
        st.dataframe(
            map_table(r, x, "Shown"),
            hide_index=True,
            width="stretch",
            column_config={"Coverage": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
        )


def page_forecasts(r: PipelineResult) -> None:
    st.title("Demand forecasts")
    ui.disclaimer()
    fe = r.forecast_eval
    zones = r.problem.zone_ids
    names = dict(zip(r.problem.zone_ids, r.problem.zone_names, strict=True))
    c1, c2 = st.columns([2, 1])
    zone = c1.selectbox(
        "Zone",
        ["__all__", *zones],
        format_func=lambda z: "All zones (system total)" if z == "__all__" else f"{z} · {names[z]}",
    )
    days = c2.slider("History shown (days)", 28, 240, 90, step=14)
    st.plotly_chart(
        charts.demand_trend(r.inputs.demand_wide(), r.forecast.daily, None if zone == "__all__" else zone, days),
        width="stretch",
    )
    st.info(
        f"Model: **{model_label(r.forecast.model)}** "
        f"({'selected by lowest validation MAE' if r.settings.model == 'auto' else 'selected by user'}). "
        + r.forecast.interval_note,
        icon=":material/info:",
    )

    t1, t2, t3, t4 = st.tabs(["Zone forecast table", "Daily forecasts", "Model evaluation", "Backtest"])
    with t1:
        tbl = r.gaps[
            [
                "zone_id",
                "zone_name",
                "forecast_demand",
                "interval_lower",
                "interval_upper",
                "stress_multiplier",
                "planning_demand",
            ]
        ]
        st.dataframe(
            tbl,
            hide_index=True,
            width="stretch",
            column_config={
                "forecast_demand": st.column_config.NumberColumn("Forecast (requests)", format="%.1f"),
                "interval_lower": st.column_config.NumberColumn("Interval low", format="%.1f"),
                "interval_upper": st.column_config.NumberColumn("Interval high", format="%.1f"),
                "stress_multiplier": st.column_config.NumberColumn("Stress ×", format="%.2f"),
                "planning_demand": st.column_config.NumberColumn(
                    "Planning demand",
                    help=f"Integer requests used by the optimizer (basis: {r.config.planning_basis})",
                ),
            },
        )
        st.caption(
            "Forecast = model output from the scenario history. Stress × = explicit scenario stress test "
            "applied after forecasting. Planning demand = what the allocator plans against."
        )
    with t2:
        daily = r.forecast.daily.pivot(index="zone_id", columns="target_date", values="predicted").round(1)
        daily.columns = [d.strftime("%a %d %b") for d in daily.columns]
        st.dataframe(daily, width="stretch")
    with t3:
        st.markdown(
            f"Chronological split by target date — train ≤ {fe.split['train_end']}, validation "
            f"{fe.split['val_start']} to {fe.split['test_start']} (exclusive), test {fe.split['test_start']} to "
            f"{fe.split['last_date']}. Evaluated on the unmodified history for horizons 1-{fe.horizon}."
        )
        a, b = st.columns(2)
        with a:
            st.markdown("**Held-out test MAE** (lower is better)")
            st.plotly_chart(charts.model_metric_chart(fe.test_metrics, "mae"), width="stretch")
        with b:
            st.markdown("**Test MAE by horizon**")
            st.plotly_chart(charts.horizon_error_chart(fe.test_per_horizon), width="stretch")
        metric_cols = {
            "mae": st.column_config.NumberColumn("MAE", format="%.2f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
            "wape": st.column_config.NumberColumn("WAPE", format="percent"),
            "bias": st.column_config.NumberColumn("Bias", format="%.2f"),
        }
        st.markdown("**Validation (used for model selection and intervals)**")
        st.dataframe(fe.validation_metrics, hide_index=True, width="stretch", column_config=metric_cols)
        st.markdown("**Test (reported only — never used for any choice)**")
        test = fe.test_metrics.assign(interval_coverage=fe.test_metrics["model"].map(fe.test_interval_coverage))
        st.dataframe(
            test,
            hide_index=True,
            width="stretch",
            column_config=metric_cols
            | {
                "interval_coverage": st.column_config.NumberColumn(
                    "Interval coverage",
                    format="percent",
                    help="Share of test actuals inside the empirical 10-90% interval (nominal 80%)",
                )
            },
        )
        best_test = fe.test_metrics.sort_values("mae").iloc[0]["model"]
        if best_test != fe.selected_model:
            st.warning(
                f"The best model on the test set ({best_test}) differs from the one selected on validation "
                f"({fe.selected_model}). Selection is not changed after seeing test results."
            )
        st.markdown("**Per-zone test MAE**")
        pz = fe.test_per_zone.pivot(index="zone_id", columns="model", values="mae").round(2)
        st.dataframe(pz, width="stretch")
        st.caption(
            "MAE = mean |forecast − actual|; RMSE = √mean squared error; WAPE = Σ|error| / Σ actual "
            "(undefined when actual demand is zero); bias = mean(forecast − actual). Units: requests/day."
        )
    with t4:
        c1, c2 = st.columns(2)
        bz = c1.selectbox("Zone", zones, format_func=lambda z: f"{z} · {names[z]}", key="bt_zone")
        bh = c2.slider("Horizon (days ahead)", 1, fe.horizon, 1, key="bt_h")
        st.plotly_chart(
            charts.backtest_chart(fe.test_predictions, bz, list(fe.test_metrics["model"]), bh), width="stretch"
        )


def constraint_checks(r: PipelineResult) -> pd.DataFrame:
    rows = []
    for key, run in r.runs.items():
        row = {"Strategy": STRATEGY_LABELS[key], "Status": run.result.status}
        if run.result.x is None:
            row.update({c: "no plan" for c in CONSTRAINTS})
        else:
            for c in CONSTRAINTS:
                if c == "coverage floor" and r.problem.coverage_floor == 0:
                    row[c] = "not active"
                    continue
                n = sum(v.constraint == c for v in run.violations)
                row[c] = "Pass" if n == 0 else f"FAIL ({n})"
            o = run.outcome
            ok = (
                (o.served <= r.problem.demand).all()
                and (o.served <= r.problem.service_cap).all()
                and (o.served <= r.problem.service_per_team * o.stationed).all()
            )
            row["service ≤ min(D, K, R·X)"] = "Pass" if ok else "FAIL"
        rows.append(row)
    return pd.DataFrame(rows)


def page_planner(r: PipelineResult) -> None:
    st.title("Allocation planner")
    ui.disclaimer()
    opt = r.runs["optimized"]
    status_strip(r)
    st.caption(
        f"Runtime {opt.result.runtime_seconds * 1000:.0f} ms · stages: "
        + ", ".join(f"{k} = {v}" for k, v in opt.result.stage_statuses.items())
        + f" · objective mode: {r.config.optimizer.mode}"
    )
    for m in opt.result.messages:
        st.caption(m)

    if opt.outcome is None or opt.violations:
        infeasible_box(r)
    else:
        cur = r.runs["current"].outcome
        o = opt.outcome
        ui.cards(
            [
                {
                    "label": "Teams moved",
                    "value": o.teams_moved,
                    "sub": f"{o.travel_minutes:.0f} team-minutes",
                    "tone": "info",
                },
                {
                    "label": "Weighted unmet",
                    "value": f"{o.weighted_unmet:,.1f}",
                    "sub": f"from {cur.weighted_unmet:,.1f}",
                    "tone": "good" if o.weighted_unmet < cur.weighted_unmet else "neutral",
                },
                {
                    "label": "Requests served",
                    "value": fmt_int(o.served.sum()),
                    "sub": f"from {cur.served.sum():,} of {r.problem.demand.sum():,}",
                    "tone": "neutral",
                },
                {
                    "label": "Unservable (above caps K)",
                    "value": fmt_int((r.problem.demand - r.problem.servable).sum()),
                    "sub": "no allocation can serve these",
                    "tone": "warning" if (r.problem.demand > r.problem.servable).any() else "neutral",
                },
            ]
        )
        st.subheader("Current vs recommended teams per zone")
        st.plotly_chart(
            charts.allocation_comparison(r.problem, r.runs["current"].result.x, opt.result.x), width="stretch"
        )

    tabs = st.tabs(["Zone gaps", "Movements", "Why this plan", "Constraint checks", "Map"])
    with tabs[0]:
        st.dataframe(
            r.gaps.drop(columns=["inaccessible"]),
            hide_index=True,
            width="stretch",
            column_config={
                "forecast_demand": st.column_config.NumberColumn("Forecast", format="%.1f"),
                "interval_lower": None,
                "interval_upper": None,
                "planning_demand": "Demand D (requests)",
                "current_teams": "Teams A",
                "current_capacity": st.column_config.NumberColumn("Capacity R·A", help="requests over horizon"),
                "service_cap": st.column_config.NumberColumn("Service cap K"),
                "staging_capacity": st.column_config.NumberColumn("Staging M (teams)"),
                "gap": st.column_config.NumberColumn("Gap D − R·A", help="> 0 shortage, < 0 spare capacity"),
                "coverage_before": st.column_config.ProgressColumn(
                    "Coverage before", format="percent", min_value=0, max_value=1
                ),
            },
        )
        st.caption(
            f"One team serves R = {r.inputs.service_rate_per_day} requests/day × {r.config.horizon_days} days = "
            f"{r.problem.service_per_team} requests over the horizon. Teams (resources) and requests "
            "(service units) are different units."
        )
    with tabs[1]:
        if opt.result.x is None:
            st.info("No movement plan: the optimizer did not return a solution.")
        else:
            mv = movements_table(r.problem, opt.result.x)
            if mv.empty:
                st.success("The optimizer recommends keeping every team in place.")
            else:
                st.dataframe(mv, hide_index=True, width="stretch")
    with tabs[2]:
        if r.rationale is None:
            st.info("No rationale: there is no validated plan.")
        else:
            st.dataframe(
                r.rationale,
                hide_index=True,
                width="stretch",
                column_config={"rationale": st.column_config.TextColumn("Rationale", width="large")},
            )
            st.caption(
                "Limiting factor = which constraint stops the zone being served further under the plan "
                "(binding constraint): demand, service cap K, staging capacity M or team capacity R·X."
            )
    with tabs[3]:
        st.dataframe(constraint_checks(r), hide_index=True, width="stretch")
        st.caption(
            "Checks are recomputed from the raw problem data by resourcemind.allocation.validation, "
            "independently of the solver. Baselines face the same checks."
        )
        for key, run in r.runs.items():
            if run.violations:
                with st.expander(f"{STRATEGY_LABELS[key]}: {len(run.violations)} violation(s)"):
                    st.dataframe(pd.DataFrame([v.__dict__ for v in run.violations]), hide_index=True, width="stretch")
    with tabs[4]:
        x, outcome, is_plan = plan_or_current(r)
        if not is_plan:
            st.caption("Showing the current allocation (no valid recommended plan).")
        st.plotly_chart(
            charts.zone_map(r.inputs.zones, r.problem, x, outcome.served, show_moves=is_plan), width="stretch"
        )


def page_lab(r: PipelineResult, cfg: AppConfig, scenario: str, model: str) -> None:
    st.title("Scenario lab")
    ui.disclaimer()
    st.markdown(
        f"Start from **{PRESETS[scenario].name}** and change inputs below. The generated dataset is never "
        "modified: overrides are stored separately and applied to copies. **Run simulation** re-validates the "
        "scenario, re-forecasts (when history changes), recomputes gaps, re-runs all three strategies and "
        "re-validates every plan."
    )
    eff = r.inputs.overrides
    z = r.inputs.zones
    defaults_w = z.set_index("zone_id")["priority_weight_default"]
    preset_inacc = set(PRESETS[scenario].overrides.inaccessible_zones)
    k = f"lab_w_{scenario}_"

    with st.form("lab_form"):
        st.markdown("##### Demand, priority and access by zone")
        editor = pd.DataFrame(
            {
                "zone_id": z["zone_id"],
                "zone_name": z["zone_name"],
                "demand_stress": [float(eff.demand_multipliers.get(zid, 1.0)) for zid in z["zone_id"]],
                "priority_weight": z["priority_weight"].astype(float),
                "inaccessible": z["inaccessible"].astype(bool),
            }
        )
        edited = st.data_editor(
            editor,
            hide_index=True,
            width="stretch",
            disabled=["zone_id", "zone_name"],
            key=k + "editor",
            column_config={
                "demand_stress": st.column_config.NumberColumn(
                    "Demand stress ×", min_value=0.0, max_value=5.0, step=0.05, help="Applied after forecasting"
                ),
                "priority_weight": st.column_config.NumberColumn(
                    "Priority weight w", min_value=0.1, max_value=5.0, step=0.05
                ),
                "inaccessible": st.column_config.CheckboxColumn("Inaccessible", help="No teams enter or leave"),
            },
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### Resources")
            resource_factor = st.slider(
                "Available teams (× current)", 0.0, 2.0, float(eff.resource_factor), 0.05, key=k + "res"
            )
            rate = st.number_input(
                "Service rate (requests / team / day)",
                1,
                40,
                int(eff.service_rate_per_day or cfg.service_rate_per_team_per_day),
                key=k + "rate",
            )
            staging = st.slider(
                "Staging capacity (× default)", 0.2, 2.0, float(eff.staging_capacity_factor), 0.05, key=k + "stage"
            )
        with c2:
            st.markdown("##### Policy")
            retain = st.number_input(
                "Minimum teams kept in each zone",
                0,
                3,
                int(cfg.min_teams_retained if eff.min_teams_retained is None else eff.min_teams_retained),
                key=k + "retain",
            )
            floor = st.slider(
                "Coverage floor per zone",
                0.0,
                0.9,
                float(eff.coverage_floor),
                0.05,
                key=k + "floor",
                help="Hard equity constraint: every zone must get at least this share of its servable demand.",
            )
        with c3:
            st.markdown("##### Travel")
            travel_mult = st.slider("All travel times ×", 0.5, 4.0, float(eff.travel_multiplier), 0.1, key=k + "tm")
            river_mult = st.slider(
                "Cross-river travel ×", 1.0, 5.0, float(eff.cross_river_multiplier), 0.1, key=k + "rm"
            )
            max_travel = st.number_input(
                "Max repositioning time (min)",
                5,
                180,
                int(eff.max_travel_minutes or cfg.max_travel_minutes),
                key=k + "mt",
            )
        pairs = [f"{a}-{b}" for a, b in itertools.combinations(r.problem.zone_ids, 2)]
        blocked = st.multiselect(
            "Blocked routes (both directions)",
            pairs,
            default=[f"{a}-{b}" for a, b in eff.blocked_routes if f"{a}-{b}" in pairs],
            key=k + "blocked",
        )
        submitted = st.form_submit_button("Run simulation", type="primary")

    if submitted:
        mult = {row.zone_id: float(row.demand_stress) for row in edited.itertuples() if float(row.demand_stress) != 1.0}
        weights = {
            row.zone_id: float(row.priority_weight)
            for row in edited.itertuples()
            if not np.isclose(float(row.priority_weight), float(defaults_w[row.zone_id]))
        }
        new = ScenarioOverrides(
            history_surge_zones=eff.history_surge_zones,
            history_surge_factor=eff.history_surge_factor,
            history_surge_days=eff.history_surge_days,
            demand_multipliers=mult,
            resource_factor=float(resource_factor),
            service_rate_per_day=None if rate == cfg.service_rate_per_team_per_day else int(rate),
            staging_capacity_factor=float(staging),
            min_teams_retained=None if retain == cfg.min_teams_retained else int(retain),
            coverage_floor=float(floor),
            priority_weights=weights,
            travel_multiplier=float(travel_mult),
            cross_river_multiplier=float(river_mult),
            blocked_routes=tuple(tuple(p.split("-")) for p in blocked),
            inaccessible_zones=tuple(row.zone_id for row in edited.itertuples() if row.inaccessible),
            max_travel_minutes=None if max_travel == cfg.max_travel_minutes else int(max_travel),
        )
        st.session_state["lab_overrides"] = new.to_dict()
        st.session_state.pop("sensitivity", None)
        st.rerun()

    st.button("Reset to preset defaults", on_click=reset_lab, key="lab_reset")
    if preset_inacc:
        st.caption(f"Preset-inaccessible zones: {', '.join(sorted(preset_inacc))} (untick to reopen them).")

    if lab_state() is None:
        st.info("No overrides yet: results equal the selected preset. Change inputs and press **Run simulation**.")
        return

    base = get_result(json.dumps(cfg.to_dict()), scenario, model, json.dumps(ScenarioOverrides().to_dict()), False)
    st.subheader("What changed")
    new_changes = [c for c in r.inputs.changes if c not in base.inputs.changes]
    removed = [c for c in base.inputs.changes if c not in r.inputs.changes]
    if not new_changes and not removed:
        st.write("Overrides match the preset: no effective change.")
    for c in new_changes:
        st.markdown(f"- **Added:** {c}")
    for c in removed:
        st.markdown(f"- **Removed (preset value reset):** {c}")

    st.subheader("Before vs after (preset → your scenario)")
    cols = [
        "status",
        "hard_violations",
        "weighted_unmet",
        "total_unmet",
        "coverage",
        "teams_moved",
        "travel_minutes",
        "min_zone_coverage",
    ]
    b = base.comparison.set_index("strategy")
    a = r.comparison.set_index("strategy")
    rows = []
    for s in ["current", "greedy", "optimized"]:
        for label, src in [("preset", b), ("your scenario", a)]:
            rows.append({"strategy": STRATEGY_LABELS[s], "run": label, **{c: src.loc[s].get(c) for c in cols}})
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "coverage": st.column_config.NumberColumn(format="percent"),
            "min_zone_coverage": st.column_config.NumberColumn("worst-zone coverage", format="percent"),
            "weighted_unmet": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    bo, ao = base.runs["optimized"].outcome, r.runs["optimized"].outcome
    if ao is None:
        infeasible_box(r)
    elif bo is not None:
        st.markdown("**Recommended teams per zone: preset vs your scenario**")
        diff = pd.DataFrame(
            {
                "zone": [f"{zid} {n}" for zid, n in zip(r.problem.zone_ids, r.problem.zone_names, strict=True)],
                "demand (preset)": base.problem.demand,
                "demand (yours)": r.problem.demand,
                "teams (preset plan)": bo.stationed,
                "teams (your plan)": ao.stationed,
                "change": ao.stationed - bo.stationed,
                "coverage (preset)": bo.served / np.maximum(base.problem.demand, 1),
                "coverage (yours)": ao.served / np.maximum(r.problem.demand, 1),
            }
        )
        st.dataframe(
            diff,
            hide_index=True,
            width="stretch",
            column_config={
                "coverage (preset)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "coverage (yours)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            },
        )


def page_strategies(r: PipelineResult, cfg: AppConfig, key_args: tuple) -> None:
    st.title("Strategy evaluation")
    ui.disclaimer()
    st.markdown(
        "All three strategies use the **same** demand, teams, capacities, travel times and eligibility rules, "
        "and are scored by the same outcome function and validator."
    )
    c = r.comparison.copy()
    c["constraint_check"] = np.where(
        c["hard_violations"] == 0, "Pass (0 violations)", "FAIL (" + c["hard_violations"].astype(str) + ")"
    )
    c.loc[c["coverage"].isna(), "constraint_check"] = "No plan produced"
    show = [
        "label",
        "status",
        "constraint_check",
        "weighted_unmet",
        "total_unmet",
        "coverage",
        "travel_minutes",
        "teams_moved",
        "min_zone_coverage",
        "coverage_range",
        "runtime_ms",
    ]
    st.dataframe(
        c[[col for col in show if col in c]],
        hide_index=True,
        width="stretch",
        column_config={
            "label": "Strategy",
            "status": "Status",
            "constraint_check": "Hard constraints",
            "weighted_unmet": st.column_config.NumberColumn("Weighted unmet", format="%.1f"),
            "total_unmet": st.column_config.NumberColumn("Unmet requests", format="%.0f"),
            "coverage": st.column_config.NumberColumn("Coverage", format="percent"),
            "travel_minutes": st.column_config.NumberColumn("Travel (team-min)", format="%.0f"),
            "teams_moved": st.column_config.NumberColumn("Teams moved", format="%.0f"),
            "min_zone_coverage": st.column_config.NumberColumn("Worst zone", format="percent"),
            "coverage_range": st.column_config.NumberColumn("Coverage range", format="percent"),
            "runtime_ms": st.column_config.NumberColumn("Runtime (ms)", format="%.1f"),
        },
    )
    a, b, d = st.columns(3)
    a.plotly_chart(charts.strategy_bars(r.comparison, "weighted_unmet", "Weighted unmet demand"), width="stretch")
    pct_cmp = (
        r.comparison.assign(coverage_pct=r.comparison["coverage"] * 100) if "coverage" in r.comparison else r.comparison
    )
    b.plotly_chart(charts.strategy_bars(pct_cmp, "coverage_pct", "Coverage (%)"), width="stretch")
    d.plotly_chart(
        charts.strategy_bars(r.comparison, "travel_minutes", "Travel (team-minutes)", ".0f"), width="stretch"
    )

    st.subheader("Fairness and distribution")
    st.markdown(
        "Priority weights do **not** guarantee fairness. Minimising weighted unmet demand can leave a single zone "
        "badly covered. Compare zone-level coverage below; the Scenario lab's coverage floor makes equity a hard "
        "constraint so the trade-off becomes explicit."
    )
    st.plotly_chart(charts.zone_coverage_chart(r.zone_tables, STRATEGY_LABELS), width="stretch")
    tiers = []
    for key, zt in r.zone_tables.items():
        t = tier_table(zt)
        t.insert(0, "strategy", STRATEGY_LABELS[key])
        tiers.append(t)
    if tiers:
        st.markdown("**Coverage by priority tier** (tiers come from the synthetic vulnerability proxy)")
        st.dataframe(
            pd.concat(tiers),
            hide_index=True,
            width="stretch",
            column_config={"coverage": st.column_config.NumberColumn(format="percent")},
        )

    st.subheader("Sensitivity analysis")
    s1, s2, s3 = st.columns([2, 2, 1])
    param = s1.selectbox("Vary", list(PARAMETERS), format_func=lambda p: PARAMETERS[p])
    lo, hi = s2.slider("Range (× current)", 0.2, 2.0, (0.5, 1.5), 0.05)
    steps = s3.number_input("Steps", 3, 15, 9)
    if st.button("Run sensitivity analysis", type="primary"):
        factors = tuple(np.round(np.linspace(lo, hi, int(steps)), 4))
        with st.spinner("Re-solving every strategy across the range..."):
            st.session_state["sensitivity"] = (param, get_sensitivity(*key_args, param, factors))
    if "sensitivity" in st.session_state:
        param_done, sweep = st.session_state["sensitivity"]
        x_label = PARAMETERS[param_done]
        g1, g2 = st.columns(2)
        g1.plotly_chart(
            charts.sensitivity_chart(sweep, "weighted_unmet", x_label, "Weighted unmet demand"), width="stretch"
        )
        g2.plotly_chart(
            charts.sensitivity_chart(
                sweep.assign(coverage=sweep["coverage"] * 100), "coverage", x_label, "Coverage (%)"
            ),
            width="stretch",
        )
        st.dataframe(
            sweep[
                [
                    "factor",
                    "label",
                    "status",
                    "hard_violations",
                    "teams_available",
                    "planning_demand",
                    "weighted_unmet",
                    "coverage",
                    "travel_minutes",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={"coverage": st.column_config.NumberColumn(format="percent")},
        )
        st.download_button(
            "Download sensitivity results (CSV)", sweep.to_csv(index=False), "resourcemind_sensitivity.csv", "text/csv"
        )

    st.subheader("Forecast error (held-out test)")
    st.dataframe(
        r.forecast_eval.test_metrics,
        hide_index=True,
        width="stretch",
        column_config={"wape": st.column_config.NumberColumn(format="percent")},
    )


def page_methodology() -> None:
    st.title("Methodology")
    ui.disclaimer()
    docs = [
        ("Mathematical model", "docs/mathematical_model.md"),
        ("Data provenance", "docs/data_dictionary.md"),
        ("Evaluation", "docs/evaluation.md"),
        ("Assumptions, limitations & ethics", "docs/assumptions_and_limitations.md"),
        ("Architecture", "docs/architecture.md"),
    ]
    tabs = st.tabs([t for t, _ in docs])
    for tab, (_, path) in zip(tabs, docs, strict=True):
        with tab:
            f = ROOT / path
            if f.exists():
                st.markdown(f.read_text(encoding="utf-8"))
            else:
                st.warning(f"{path} not found.")


# --------------------------------------------------------------------------- main
PAGE_SLUGS = {
    "overview": "Command overview",
    "forecasts": "Demand forecasts",
    "planner": "Allocation planner",
    "lab": "Scenario lab",
    "strategies": "Strategy evaluation",
    "methodology": "Methodology",
}


def apply_deep_link() -> None:
    """On first load, honour ?page=<slug>&scenario=<key> so views can be linked directly."""
    if st.session_state.get("deep_link_done"):
        return
    st.session_state["deep_link_done"] = True
    params = st.query_params
    if params.get("page") in PAGE_SLUGS:
        st.session_state["page"] = PAGE_SLUGS[params["page"]]
    if params.get("scenario") in PRESETS:
        st.session_state["scenario"] = params["scenario"]


def main() -> None:
    st.session_state.setdefault("lab_overrides", None)
    apply_deep_link()
    page, cfg, scenario, model = sidebar()
    overrides = lab_state()
    replace = overrides is not None
    key_args = (
        json.dumps(cfg.to_dict()),
        scenario,
        model,
        json.dumps(overrides if replace else ScenarioOverrides().to_dict()),
        replace,
    )
    if page == "Methodology":
        page_methodology()
        return
    try:
        with st.spinner("Forecasting demand, solving allocation and validating plans..."):
            result = get_result(*key_args)
    except DataValidationError as exc:
        st.error(f"Scenario failed validation: {exc}")
        st.button("Reset to preset defaults", on_click=reset_lab)
        return
    export_panel(result)
    if page == "Command overview":
        page_overview(result)
    elif page == "Demand forecasts":
        page_forecasts(result)
    elif page == "Allocation planner":
        page_planner(result)
    elif page == "Scenario lab":
        page_lab(result, cfg, scenario, model)
    elif page == "Strategy evaluation":
        page_strategies(result, cfg, key_args)


main()
