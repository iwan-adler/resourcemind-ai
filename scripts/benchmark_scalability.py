"""Scalability benchmark: solve time and solution quality as the number of zones grows.

Usage:  python scripts/benchmark_scalability.py [--sizes 10 25 50 100] [--seeds 42 7 2024]
                                                [--time-limit 20] [--workers 1] [--out results]

Problems come from ``resourcemind.simulation.scaled`` (same structure as the demo
data at a larger scale). For each size and seed it runs greedy and the CP-SAT
optimizer in both planning modes (the day-by-day optimizer warm-started from the
greedy plan, as in the app) and writes ``results/scalability.csv``.
``time-limit`` applies to each optimizer stage. Runtimes depend on hardware.
"""

import argparse
import time
from pathlib import Path

import _path  # noqa: F401
import pandas as pd

from resourcemind.allocation.baselines import greedy_allocation, greedy_daily
from resourcemind.allocation.models import compute_outcome
from resourcemind.allocation.multiperiod import compute_daily_outcome, optimize_daily
from resourcemind.allocation.optimizer import optimize_allocation
from resourcemind.allocation.validation import validate_allocation, validate_daily_plan
from resourcemind.config import RESULTS_DIR
from resourcemind.simulation.scaled import scaled_problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1, help="CP-SAT workers (1 = deterministic, as in the app)")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    rows = []
    for n in args.sizes:
        for seed in args.seeds:
            single, daily = scaled_problems(n, seed=seed)
            base = {
                "zones": n,
                "seed": seed,
                "teams": int(single.available.sum()),
                "eligible_routes": int(single.eligible.sum() - n),
            }
            t = time.perf_counter()
            g = greedy_allocation(single)
            g_time = time.perf_counter() - t
            g_wu = compute_outcome(single, g.x).weighted_unmet
            o = optimize_allocation(single, time_limit_seconds=args.time_limit, num_workers=args.workers, seed=seed)
            o_wu = compute_outcome(single, o.x).weighted_unmet if o.x is not None else None
            rows.append(
                base
                | {
                    "mode": "single",
                    "greedy_seconds": g_time,
                    "greedy_weighted_unmet": g_wu,
                    "optimizer_status": o.status,
                    "optimizer_seconds": o.runtime_seconds,
                    "optimizer_weighted_unmet": o_wu,
                    "lower_bound": o.diagnostics.get("weighted_unmet_lower_bound"),
                    "gap": o.diagnostics.get("weighted_unmet_gap"),
                    "violations": len(validate_allocation(single, o.x)),
                }
            )
            print(rows[-1], flush=True)

            t = time.perf_counter()
            gd = greedy_daily(daily)
            gd_time = time.perf_counter() - t
            gd_wu = compute_daily_outcome(daily, gd.plan).weighted_unmet
            od = optimize_daily(
                daily, time_limit_seconds=args.time_limit, num_workers=args.workers, seed=seed, hint=gd.plan
            )
            od_wu = compute_daily_outcome(daily, od.plan).weighted_unmet if od.plan is not None else None
            rows.append(
                base
                | {
                    "mode": "daily",
                    "greedy_seconds": gd_time,
                    "greedy_weighted_unmet": gd_wu,
                    "optimizer_status": od.status,
                    "optimizer_seconds": od.runtime_seconds,
                    "optimizer_weighted_unmet": od_wu,
                    "lower_bound": od.diagnostics.get("weighted_unmet_lower_bound"),
                    "gap": od.diagnostics.get("weighted_unmet_gap"),
                    "violations": len(validate_daily_plan(daily, od.plan)),
                }
            )
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "scalability.csv" if args.workers == 1 else f"scalability_workers{args.workers}.csv"
    df.to_csv(out / name, index=False)
    summary = df.groupby(["mode", "zones"]).agg(
        teams=("teams", "mean"),
        greedy_s=("greedy_seconds", "mean"),
        opt_s_mean=("optimizer_seconds", "mean"),
        opt_s_max=("optimizer_seconds", "max"),
        optimal=("optimizer_status", lambda s: int((s == "OPTIMAL").sum())),
        runs=("optimizer_status", "size"),
        max_gap=("gap", "max"),
        violations=("violations", "sum"),
    )
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print(f"Wrote {out / name}")


if __name__ == "__main__":
    main()
