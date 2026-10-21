"""Reproducible evaluation across scenarios and random seeds.

Usage:  python scripts/run_evaluation.py [--seeds 42 7 2024 99 314] [--out results]

Writes:
  results/evaluation_results.json   full machine-readable record
  results/forecast_metrics.csv      held-out test metrics per seed and model
  results/strategy_comparison.csv   allocation metrics per seed, case, strategy
  results/summary_by_scenario.csv   mean over seeds per case and strategy
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
from resourcemind.simulation.scenarios import PRESETS, ScenarioOverrides

# Extra evaluation case: the shortage preset with a 30% per-zone coverage floor.
CASES = [(k, k, ScenarioOverrides()) for k in PRESETS] + [
    ("shortage_floor30", "shortage", ScenarioOverrides(coverage_floor=0.3))
]
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

    forecast_rows, strategy_rows, runs = [], [], []
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
        for case, preset, overrides in CASES:
            t = time.perf_counter()
            result = run_pipeline(
                config,
                PipelineSettings(scenario=preset, overrides=overrides),
                dataset=dataset,
                forecast_eval=fe,
            )
            end_to_end = time.perf_counter() - t
            for rec in result.comparison.to_dict(orient="records"):
                strategy_rows.append(
                    {
                        "seed": seed,
                        "case": case,
                        "forecast_model": result.forecast.model,
                        "end_to_end_seconds": end_to_end,
                        **rec,
                    }
                )
            opt = result.runs["optimized"].result
            runs.append(
                {
                    "seed": seed,
                    "case": case,
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

    forecast_df.to_csv(out / "forecast_metrics.csv", index=False)
    strategy_df.to_csv(out / "strategy_comparison.csv", index=False)
    summary.to_csv(out / "summary_by_scenario.csv", index=False)
    record = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_provenance": "synthetic",
        "seeds": args.seeds,
        "config": base.to_dict() | {"seed": "varies (see seeds)"},
        "forecast_test_mean_over_seeds": forecast_summary.reset_index().to_dict(orient="records"),
        "forecast_model_selected_on_validation": selected_counts,
        "allocation_mean_over_seeds": summary.replace({np.nan: None}).to_dict(orient="records"),
        "optimizer_status_counts": status_counts.to_dict(orient="records"),
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
    print("\nOptimizer status counts:")
    print(status_counts.to_string(index=False))
    print(f"\nWrote results to {out.resolve()} in {record['total_runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
