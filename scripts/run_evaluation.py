"""Reproducible evaluation across scenarios and random seeds.

Usage:  python scripts/run_evaluation.py [--seeds 42 7 2024 99 314] [--out results]

Writes:
  results/evaluation_results.json   full machine-readable record
  results/forecast_metrics.csv      held-out test metrics per seed and model
  results/strategy_comparison.csv   allocation metrics per seed, case, strategy
  results/summary_by_scenario.csv   mean over seeds per case and strategy
  results/robustness.csv            plans scored on held-out forecast errors (single-period presets)

Cases ending in "_daily" use the day-by-day (multi-period) planning mode on the
same scenario; the others use the default single-period mode.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd

from resourcemind.config import RESULTS_DIR, load_config
from resourcemind.data.generate import generate_dataset
from resourcemind.forecasting.evaluation import evaluate_forecasters
from resourcemind.pipeline import PipelineSettings, run_pipeline
from resourcemind.simulation.robustness import robustness_check
from resourcemind.simulation.scenarios import PRESETS, ScenarioOverrides

# (case, preset, overrides, planning_mode, forecast model). Extra case: the shortage
# preset with a 30% per-zone coverage floor. Every case runs in both planning modes
# with the auto-selected forecast. The auto choice is usually the 28-day moving
# average, whose daily forecast is flat, so the "_daily_seasonal" cases repeat the
# presets with the weekday-varying 3-week seasonal average to test whether the
# day-by-day model gains when demand changes from day to day.
_BASE_CASES = [(k, k, ScenarioOverrides()) for k in PRESETS] + [
    ("shortage_floor30", "shortage", ScenarioOverrides(coverage_floor=0.3))
]
CASES = (
    [(c, p, o, "single", "auto") for c, p, o in _BASE_CASES]
    + [(f"{c}_daily", p, o, "daily", "auto") for c, p, o in _BASE_CASES]
    + [(f"{k}_daily_seasonal", k, ScenarioOverrides(), "daily", "seasonal_average_3w") for k in PRESETS]
)
SUMMARY_COLUMNS = [
    "total_unmet",
    "weighted_unmet",
    "coverage",
    "travel_minutes",
    "teams_moved",
    "hard_violations",
    "runtime_ms",
    "min_zone_coverage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024, 99, 314])
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_config()

    forecast_rows, strategy_rows, runs, robust_rows = [], [], [], []
    started = time.perf_counter()
    for seed in args.seeds:
        config = base.with_updates(seed=seed)
        dataset = generate_dataset(config)
        t = time.perf_counter()
        fe = evaluate_forecasters(dataset.demand_wide(), config.horizon_days, config.forecast, random_state=seed)
        fe_runtime = time.perf_counter() - t
        val = fe.validation_metrics.set_index("model")
        for rec in fe.test_metrics.to_dict(orient="records"):
            forecast_rows.append(
                {
                    "seed": seed,
                    **rec,
                    "validation_mae": float(val.loc[rec["model"], "mae"]),
                    "selected_on_validation": rec["model"] == fe.selected_model,
                    "test_interval_coverage": fe.test_interval_coverage[rec["model"]],
                }
            )
        for case, preset, overrides, mode, model in CASES:
            t = time.perf_counter()
            result = run_pipeline(
                config.with_updates(planning_mode=mode),
                PipelineSettings(scenario=preset, model=model, overrides=overrides),
                dataset=dataset,
                forecast_eval=fe,
            )
            end_to_end = time.perf_counter() - t
            for rec in result.active_comparison.to_dict(orient="records"):
                strategy_rows.append(
                    {
                        "seed": seed,
                        "case": case,
                        "planning_mode": mode,
                        "forecast_model": result.forecast.model,
                        "end_to_end_seconds": end_to_end,
                        **rec,
                    }
                )
            if mode == "single" and case in PRESETS:
                rob = robustness_check(result, n_samples=200, seed=seed, include_expected=True)
                for rec in rob.summary.to_dict(orient="records"):
                    robust_rows.append({"seed": seed, "case": case, "notes": " | ".join(rob.notes), **rec})
            opt = result.daily.runs["optimized"].result if result.daily is not None else result.runs["optimized"].result
            runs.append(
                {
                    "seed": seed,
                    "case": case,
                    "planning_mode": mode,
                    "forecast_evaluation_seconds": fe_runtime,
                    "end_to_end_seconds_excluding_forecast_eval": end_to_end,
                    "changes": result.inputs.changes,
                    "optimizer_status": opt.status,
                    "optimizer_stage_statuses": opt.stage_statuses,
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    strategy_df = pd.DataFrame(strategy_rows)
    summary = strategy_df.groupby(["case", "strategy"], sort=False)[SUMMARY_COLUMNS].mean().reset_index()
    forecast_summary = forecast_df.groupby("model")[["mae", "rmse", "wape", "test_interval_coverage"]].mean()
    selected_counts = forecast_df[forecast_df["selected_on_validation"]]["model"].value_counts().to_dict()
    status_counts = (
        strategy_df[strategy_df["strategy"] == "optimized"]
        .groupby(["case", "status"])
        .size()
        .rename("runs")
        .reset_index()
    )
    # Head-to-head per run: does the optimizer beat greedy on weighted unmet demand / travel?
    wide = strategy_df.pivot_table(
        index=["case", "seed"], columns="strategy", values=["weighted_unmet", "travel_minutes"]
    )
    diff = wide["weighted_unmet"]["optimized"] - wide["weighted_unmet"]["greedy"]
    tdiff = wide["travel_minutes"]["optimized"] - wide["travel_minutes"]["greedy"]
    head_to_head = (
        pd.DataFrame(
            {
                "optimizer_better": diff < -1e-9,
                "tie": diff.abs() <= 1e-9,
                "optimizer_worse": diff > 1e-9,
                "weighted_unmet_gap": -diff,
                "tie_and_less_travel": (diff.abs() <= 1e-9) & (tdiff < -1e-9),
            }
        )
        .groupby(level="case", sort=False)
        .agg(
            runs=("tie", "size"),
            optimizer_better=("optimizer_better", "sum"),
            tie=("tie", "sum"),
            optimizer_worse=("optimizer_worse", "sum"),
            tie_and_less_travel=("tie_and_less_travel", "sum"),
            mean_weighted_unmet_gap=("weighted_unmet_gap", "mean"),
            max_weighted_unmet_gap=("weighted_unmet_gap", "max"),
        )
        .reindex([c for c, *_ in CASES])
        .reset_index()
    )

    forecast_df.to_csv(out / "forecast_metrics.csv", index=False)
    strategy_df.to_csv(out / "strategy_comparison.csv", index=False)
    summary.to_csv(out / "summary_by_scenario.csv", index=False)
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(out / "robustness.csv", index=False)
    robust_cols = ["mean_coverage", "p05_coverage", "prob_below_target", "mean_weighted_unmet", "mean_worst_zone"]
    robust_summary = robust_df.groupby(["case", "strategy"], sort=False)[robust_cols].mean().reset_index()
    record = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_provenance": "synthetic",
        "seeds": args.seeds,
        "config": base.to_dict() | {"seed": "varies (see seeds)"},
        "forecast_test_mean_over_seeds": forecast_summary.reset_index().to_dict(orient="records"),
        "forecast_model_selected_on_validation": selected_counts,
        "allocation_mean_over_seeds": summary.replace({np.nan: None}).to_dict(orient="records"),
        "optimizer_status_counts": status_counts.to_dict(orient="records"),
        "optimizer_vs_greedy": head_to_head.to_dict(orient="records"),
        "robustness_mean_over_seeds": robust_summary.to_dict(orient="records"),
        "runs": runs,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (out / "evaluation_results.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("Forecast test metrics (mean over seeds):")
    print(forecast_summary.round(3).to_string())
    print("Model selected on validation:", selected_counts)
    print("\nAllocation (mean over seeds):")
    print(summary.round(3).to_string(index=False))
    print("\nOptimizer vs greedy (weighted unmet demand, per run):")
    print(head_to_head.round(3).to_string(index=False))
    print("\nRobustness on held-out forecast errors (mean over seeds):")
    print(robust_summary.round(3).to_string(index=False))
    print("\nOptimizer status counts:")
    print(status_counts.to_string(index=False))
    print(f"\nWrote results to {out.resolve()} in {record['total_runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
