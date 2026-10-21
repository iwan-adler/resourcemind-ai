"""End-to-end workflow: data -> scenario -> forecast -> gaps -> allocation -> evaluation.

Used by the Streamlit app, the evaluation command and the integration tests, so
every surface runs exactly the same computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from resourcemind.allocation.baselines import current_allocation, greedy_allocation
from resourcemind.allocation.explain import plan_summary, zone_rationale
from resourcemind.allocation.models import AllocationProblem, AllocationResult, Outcome, compute_outcome
from resourcemind.allocation.optimizer import optimize_allocation
from resourcemind.allocation.validation import Violation, validate_allocation
from resourcemind.config import AppConfig
from resourcemind.data.generate import Dataset, generate_dataset
from resourcemind.data.validate import DataValidationError, validate_dataset
from resourcemind.forecasting.evaluation import (
    ForecastEvaluation,
    ZoneForecast,
    evaluate_forecasters,
    forecast_future,
)
from resourcemind.metrics.evaluation import coverage_ratio, fairness_summary, zone_outcome_table
from resourcemind.simulation.scenarios import (
    ScenarioInputs,
    ScenarioOverrides,
    apply_scenario,
    build_problem,
    planning_demand,
)

STRATEGIES = ["current", "greedy", "optimized"]
STRATEGY_LABELS = {
    "current": "Current allocation",
    "greedy": "Greedy baseline",
    "optimized": "Optimized (CP-SAT)",
}


@dataclass(frozen=True)
class PipelineSettings:
    scenario: str = "normal"
    model: str = "auto"  # "auto" = best validation MAE
    overrides: ScenarioOverrides = field(default_factory=ScenarioOverrides)
    # True: overrides fully replace the preset's own overrides (Scenario Lab).
    replace_preset: bool = False


@dataclass
class StrategyRun:
    result: AllocationResult
    outcome: Outcome | None
    violations: list[Violation]


@dataclass
class PipelineResult:
    config: AppConfig
    settings: PipelineSettings
    inputs: ScenarioInputs
    forecast_eval: ForecastEvaluation
    forecast: ZoneForecast
    demand: pd.DataFrame
    gaps: pd.DataFrame
    problem: AllocationProblem
    runs: dict[str, StrategyRun]
    comparison: pd.DataFrame
    zone_tables: dict[str, pd.DataFrame]
    rationale: pd.DataFrame | None
    summary_lines: list[str]
    timings: dict[str, float]
    created_at: str


def gap_table(inputs: ScenarioInputs, demand: pd.DataFrame, problem: AllocationProblem) -> pd.DataFrame:
    z = inputs.zones.reset_index(drop=True)
    capacity = problem.available * problem.service_per_team
    served = np.minimum(problem.servable, capacity)
    gap = problem.demand - capacity
    status = np.where(gap > 0, "shortage", np.where(gap < -problem.service_per_team, "surplus", "balanced"))
    return pd.DataFrame(
        {
            "zone_id": z["zone_id"],
            "zone_name": z["zone_name"],
            "priority_tier": z["priority_tier"],
            "priority_weight": z["priority_weight"],
            "forecast_demand": demand["forecast_total"].round(1),
            "interval_lower": demand["lower_total"].round(1),
            "interval_upper": demand["upper_total"].round(1),
            "stress_multiplier": demand["stress_multiplier"],
            "planning_demand": problem.demand,
            "current_teams": problem.available,
            "current_capacity": capacity,
            "service_cap": problem.service_cap,
            "staging_capacity": problem.staging_capacity,
            "gap": gap,
            "status": status,
            "coverage_before": [coverage_ratio(s, d) for s, d in zip(served, problem.demand, strict=True)],
            "inaccessible": z["inaccessible"],
        }
    )


def run_strategies(problem: AllocationProblem, config: AppConfig) -> dict[str, StrategyRun]:
    opt = config.optimizer
    results = {
        "current": current_allocation(problem),
        "greedy": greedy_allocation(problem),
        "optimized": optimize_allocation(
            problem,
            mode=opt.mode,
            travel_penalty_lambda=opt.travel_penalty_lambda,
            time_limit_seconds=opt.time_limit_seconds,
            num_workers=opt.num_workers,
            seed=config.seed,
        ),
    }
    runs = {}
    for key, res in results.items():
        outcome = compute_outcome(problem, res.x) if res.has_solution else None
        violations = validate_allocation(problem, res.x)
        if key == "optimized" and violations and res.status in ("OPTIMAL", "FEASIBLE"):
            # Never present a constraint-violating plan as feasible.
            res.messages.append("Independent validation FAILED; plan rejected.")
            res.status = "INVALID"
        runs[key] = StrategyRun(res, outcome, violations)
    return runs


def comparison_table(problem: AllocationProblem, runs: dict[str, StrategyRun]) -> pd.DataFrame:
    rows = []
    total_demand = float(problem.demand.sum())
    for key in STRATEGIES:
        run = runs[key]
        row = {
            "strategy": key,
            "label": STRATEGY_LABELS[key],
            "status": run.result.status,
            "hard_violations": len(run.violations),
            "runtime_ms": run.result.runtime_seconds * 1000.0,
        }
        if run.outcome is not None:
            o = run.outcome
            zt = zone_outcome_table(
                problem.zone_ids,
                problem.zone_names,
                problem.priority_tiers,
                problem.weights,
                problem.demand,
                o.served,
            )
            row.update(
                total_demand=total_demand,
                served=float(o.served.sum()),
                total_unmet=float(o.unmet.sum()),
                weighted_unmet=o.weighted_unmet,
                coverage=coverage_ratio(o.served.sum(), total_demand),
                travel_minutes=o.travel_minutes,
                teams_moved=o.teams_moved,
                **fairness_summary(zt),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_pipeline(
    config: AppConfig,
    settings: PipelineSettings | None = None,
    dataset: Dataset | None = None,
    forecast_eval: ForecastEvaluation | None = None,
) -> PipelineResult:
    settings = settings or PipelineSettings()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    dataset = dataset or generate_dataset(config)
    validate_dataset(dataset).raise_if_failed()
    timings["data"] = time.perf_counter() - t0

    t = time.perf_counter()
    inputs = apply_scenario(dataset, config, settings.scenario, settings.overrides, settings.replace_preset)
    if not inputs.validation.ok:
        raise DataValidationError("; ".join(inputs.validation.errors))
    timings["scenario"] = time.perf_counter() - t

    t = time.perf_counter()
    if forecast_eval is None or forecast_eval.horizon != config.horizon_days:
        # Accuracy is measured on the unmodified history (scenario-independent).
        forecast_eval = evaluate_forecasters(
            dataset.demand_wide(), config.horizon_days, config.forecast, random_state=config.seed
        )
    timings["forecast_evaluation"] = time.perf_counter() - t

    t = time.perf_counter()
    model = forecast_eval.selected_model if settings.model == "auto" else settings.model
    forecast = forecast_future(
        inputs.demand_wide(),
        model,
        config.horizon_days,
        forecast_eval.interval_quantiles[model],
        lookback=config.forecast.lookback_days,
        random_state=config.seed,
    )
    timings["forecast"] = time.perf_counter() - t

    demand = planning_demand(inputs, forecast.totals, config.planning_basis)
    problem = build_problem(inputs, demand["planning_demand"].to_numpy())
    gaps = gap_table(inputs, demand, problem)

    t = time.perf_counter()
    runs = run_strategies(problem, config)
    timings["allocation"] = time.perf_counter() - t

    comparison = comparison_table(problem, runs)
    zone_tables = {
        k: zone_outcome_table(
            problem.zone_ids,
            problem.zone_names,
            problem.priority_tiers,
            problem.weights,
            problem.demand,
            r.outcome.served,
        )
        for k, r in runs.items()
        if r.outcome is not None
    }
    before = runs["current"].outcome
    opt = runs["optimized"]
    if opt.outcome is not None and not opt.violations:
        rationale = zone_rationale(problem, opt.result.x, before, opt.outcome)
        summary = plan_summary(problem, before, opt.outcome)
    else:
        rationale, summary = (
            None,
            list(opt.result.messages) + list(opt.result.diagnostics.get("infeasibility_reasons", [])),
        )
    timings["total"] = time.perf_counter() - t0
    return PipelineResult(
        config=config,
        settings=settings,
        inputs=inputs,
        forecast_eval=forecast_eval,
        forecast=forecast,
        demand=demand,
        gaps=gaps,
        problem=problem,
        runs=runs,
        comparison=comparison,
        zone_tables=zone_tables,
        rationale=rationale,
        summary_lines=summary,
        timings=timings,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def run_record(result: PipelineResult) -> dict:
    """Machine-readable record of one run (exported as JSON)."""
    cfg = result.config
    opt = result.runs["optimized"].result
    return {
        "created_at_utc": result.created_at,
        "data_provenance": "synthetic",
        "scenario": {
            "key": result.inputs.key,
            "name": result.inputs.name,
            "changes": result.inputs.changes,
            "overrides": result.inputs.overrides.to_dict(),
        },
        "seed": cfg.seed,
        "forecast": {
            "model": result.forecast.model,
            "selection": "best validation MAE" if result.settings.model == "auto" else "user selected",
            "horizon_days": cfg.horizon_days,
            "planning_basis": cfg.planning_basis,
            "interval_note": result.forecast.interval_note,
            "test_metrics": result.forecast_eval.test_metrics.to_dict(orient="records"),
        },
        "resources": {
            "teams_available": int(result.problem.available.sum()),
            "service_requests_per_team_per_day": result.inputs.service_rate_per_day,
            "max_travel_minutes": result.inputs.max_travel_minutes,
            "min_teams_retained": result.inputs.min_teams_retained,
        },
        "solver": {
            "name": "OR-Tools CP-SAT",
            "mode": cfg.optimizer.mode,
            "travel_penalty_lambda": cfg.optimizer.travel_penalty_lambda,
            "status": opt.status,
            "stage_statuses": opt.stage_statuses,
            "time_limit_seconds": cfg.optimizer.time_limit_seconds,
        },
        "strategies": result.comparison.replace({np.nan: None}).to_dict(orient="records"),
        "timings_seconds": result.timings,
    }


def zone_plan_table(result: PipelineResult) -> pd.DataFrame:
    """Zone-level allocation recommendation (exported as CSV)."""
    p = result.problem
    cur = result.runs["current"].outcome
    opt = result.runs["optimized"].outcome
    table = pd.DataFrame(
        {
            "zone_id": p.zone_ids,
            "zone_name": p.zone_names,
            "priority_tier": p.priority_tiers,
            "priority_weight": p.weights,
            "planning_demand_requests": p.demand,
            "teams_current": cur.stationed,
            "served_current": cur.served,
            "coverage_current": [coverage_ratio(s, d) for s, d in zip(cur.served, p.demand, strict=True)],
        }
    )
    if opt is not None:
        table["teams_recommended"] = opt.stationed
        table["served_recommended"] = opt.served
        table["unmet_recommended"] = opt.unmet
        table["coverage_recommended"] = [coverage_ratio(s, d) for s, d in zip(opt.served, p.demand, strict=True)]
        table["limiting_factor"] = opt.limiting_factor
    table["solver_status"] = result.runs["optimized"].result.status
    return table


def run_summary_markdown(result: PipelineResult) -> str:
    """Human-readable run summary for download."""
    rec = run_record(result)
    lines = [
        "# ResourceMind AI - run summary",
        "",
        "> Decision-support prototype on SYNTHETIC data. Not an operational dispatch instruction.",
        "",
        f"- Generated (UTC): {rec['created_at_utc']}",
        f"- Scenario: {rec['scenario']['name']} (`{rec['scenario']['key']}`)",
        f"- Changes: {'; '.join(rec['scenario']['changes']) or 'none'}",
        f"- Seed: {rec['seed']}",
        f"- Forecast model: {rec['forecast']['model']} ({rec['forecast']['selection']}), "
        f"horizon {rec['forecast']['horizon_days']} days, planning basis `{rec['forecast']['planning_basis']}`",
        f"- Solver: {rec['solver']['name']}, mode `{rec['solver']['mode']}`, status **{rec['solver']['status']}**",
        f"- Teams available: {rec['resources']['teams_available']}; "
        f"{rec['resources']['service_requests_per_team_per_day']} requests/team/day",
        "",
        "## Plan summary",
        "",
        *[f"- {line}" for line in result.summary_lines],
        "",
        "## Strategy comparison",
        "",
        "| Strategy | Status | Weighted unmet | Unmet | Coverage | Travel (team-min) | Moved | Violations |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result.comparison.to_dict(orient="records"):
        if row.get("coverage") is None or pd.isna(row.get("coverage")):
            lines.append(f"| {row['label']} | {row['status']} | - | - | - | - | - | {row['hard_violations']} |")
        else:
            lines.append(
                f"| {row['label']} | {row['status']} | {row['weighted_unmet']:.1f} | {row['total_unmet']:.0f} | "
                f"{row['coverage']:.1%} | {row['travel_minutes']:.0f} | {row['teams_moved']:.0f} | "
                f"{row['hard_violations']} |"
            )
    lines += [
        "",
        "## Forecast test metrics (held-out, chronological)",
        "",
        "| Model | MAE | RMSE | WAPE |",
        "|---|---|---|---|",
    ]
    for m in rec["forecast"]["test_metrics"]:
        lines.append(f"| {m['model']} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['wape']:.1%} |")
    return "\n".join(lines) + "\n"
