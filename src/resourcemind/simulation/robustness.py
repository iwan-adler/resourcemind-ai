"""Robustness of allocation plans to forecast error (single-period plans).

A plan is made BEFORE demand is known. This module asks how each fixed plan would
have performed across many plausible demand outcomes:

    realised_i = round(forecast_i x stress_i x (1 + e_i)),   e = resampled forecast errors

The errors come from the forecasting model's own relative errors on horizon
totals. Each sample takes one whole forecast window, i.e. the errors of every zone
from the same day, so errors that hit many zones at once stay correlated.

Out-of-sample by construction:
* plans that USE errors (the expected-outcome plan) are built from VALIDATION-window
  errors only;
* every plan is SCORED on errors from the held-out TEST window, which no plan saw.
Each window holds about 20 complete forecast windows, so samples repeat them; this
measures behaviour under errors the model actually made, not a full probability model.

Strategies compared on the same samples: current allocation, greedy, the optimized
plan for the point forecast, the optimized plan for the upper empirical interval
("cautious"), and the optimized plan for the EXPECTED outcome over sampled demand.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from resourcemind.allocation.models import AllocationProblem
from resourcemind.allocation.optimizer import optimize_allocation, optimize_expected

ROBUST_LABELS = {
    "current": "Current allocation",
    "greedy": "Greedy baseline",
    "optimized_point": "Optimized, point forecast",
    "optimized_upper": "Optimized, upper forecast (cautious)",
    "optimized_expected": "Optimized, expected outcome",
}
COVERAGE_TARGET = 0.90
FIT_SAMPLES = 40  # demand samples the expected-outcome plan is optimised on


@dataclass
class RobustnessResult:
    n_samples: int
    n_test_windows: int
    n_fit_windows: int
    model: str
    summary: pd.DataFrame  # one row per strategy
    coverage_samples: pd.DataFrame  # strategy, label, sample, coverage, weighted_unmet
    notes: list[str]


def evaluate_plan_on_samples(problem: AllocationProblem, x: np.ndarray, realised: np.ndarray) -> dict[str, np.ndarray]:
    """Service of a FIXED plan under each sampled demand vector (rows of ``realised``)."""
    stationed = np.asarray(x).sum(axis=0)
    capacity = problem.service_per_team * stationed
    served = np.minimum(np.minimum(realised, problem.service_cap[None, :]), capacity[None, :])
    unmet = realised - served
    totals = realised.sum(axis=1)
    coverage = np.where(totals > 0, served.sum(axis=1) / np.maximum(totals, 1), 1.0)
    zone_cov = np.where(realised > 0, served / np.maximum(realised, 1), 1.0)
    return {
        "coverage": coverage,
        "weighted_unmet": (unmet * problem.weights[None, :]).sum(axis=1),
        "worst_zone": zone_cov.min(axis=1),
    }


def sample_demand(
    forecast_total: np.ndarray, stress: np.ndarray, residuals: pd.DataFrame | None, n_samples: int, seed: int
) -> np.ndarray:
    """(n_samples x zones) demand outcomes by resampling whole windows of relative errors."""
    if residuals is None or residuals.empty:
        raise ValueError("no forecast residuals available for this model")
    rng = np.random.default_rng(seed)
    rel = residuals.to_numpy(dtype=float)[rng.integers(0, len(residuals), n_samples)]
    base = np.asarray(forecast_total, dtype=float) * np.asarray(stress, dtype=float)
    return np.maximum(np.round(base[None, :] * (1.0 + rel)), 0).astype(int)


def robustness_check(result, n_samples: int = 200, seed: int = 0, include_expected: bool = False) -> RobustnessResult:
    """Run the check for a ``PipelineResult`` (its single-period plans).

    ``include_expected`` adds the expected-outcome plan (slower: a larger CP-SAT model).
    """
    p: AllocationProblem = result.problem
    model = result.forecast.model
    fe = result.forecast_eval
    fit_residuals = fe.validation_total_residuals.get(model)
    test_residuals = fe.test_total_residuals.get(model)
    stress = result.demand["stress_multiplier"].to_numpy(dtype=float)
    forecast = result.demand["forecast_total"].to_numpy(dtype=float)
    realised = sample_demand(forecast, stress, test_residuals, n_samples, seed)

    opt_cfg = result.config.optimizer
    solve = {
        "time_limit_seconds": opt_cfg.time_limit_seconds,
        "num_workers": opt_cfg.num_workers,
        "seed": result.config.seed,
    }
    notes: list[str] = []
    plans: dict[str, np.ndarray] = {
        "current": result.runs["current"].result.x,
        "greedy": result.runs["greedy"].result.x,
    }
    for key, column in [("optimized_point", "forecast_total"), ("optimized_upper", "upper_total")]:
        demand = np.round(result.demand[column].to_numpy(dtype=float) * stress).astype(int)
        res = optimize_allocation(
            replace(p, demand=demand), mode=opt_cfg.mode, travel_penalty_lambda=opt_cfg.travel_penalty_lambda, **solve
        )
        if res.x is None:
            notes.append(f"{ROBUST_LABELS[key]}: no plan ({res.status}); left out.")
        else:
            plans[key] = res.x
    if include_expected:
        fit = sample_demand(forecast, stress, fit_residuals, FIT_SAMPLES, seed + 1)
        expected = optimize_expected(p, fit, **solve)
        if expected.x is None:
            notes.append(f"{ROBUST_LABELS['optimized_expected']}: no plan ({expected.status}); left out.")
        else:
            plans["optimized_expected"] = expected.x
            if expected.status != "OPTIMAL":
                notes.append(f"{ROBUST_LABELS['optimized_expected']}: {expected.status} within the time limit.")

    rows, frames = [], []
    for key, x in plans.items():
        m = evaluate_plan_on_samples(p, x, realised)
        rows.append(
            {
                "strategy": key,
                "label": ROBUST_LABELS[key],
                "mean_coverage": float(m["coverage"].mean()),
                "p05_coverage": float(np.quantile(m["coverage"], 0.05)),
                "prob_below_target": float((m["coverage"] < COVERAGE_TARGET).mean()),
                "mean_weighted_unmet": float(m["weighted_unmet"].mean()),
                "p95_weighted_unmet": float(np.quantile(m["weighted_unmet"], 0.95)),
                "mean_worst_zone": float(m["worst_zone"].mean()),
                "teams_moved": int((np.asarray(x) * (1 - np.eye(p.n, dtype=int))).sum()),
            }
        )
        frames.append(
            pd.DataFrame(
                {
                    "strategy": key,
                    "label": ROBUST_LABELS[key],
                    "sample": np.arange(n_samples),
                    "coverage": m["coverage"],
                    "weighted_unmet": m["weighted_unmet"],
                }
            )
        )
    note = (
        f"{n_samples} outcomes resampled from {len(test_residuals)} held-out test forecast windows of the {model} model"
    )
    if include_expected:
        note += (
            f"; the expected-outcome plan was built from {FIT_SAMPLES} samples of {len(fit_residuals)} "
            "validation windows only"
        )
    notes.append(note + ".")
    return RobustnessResult(
        n_samples=n_samples,
        n_test_windows=len(test_residuals),
        n_fit_windows=len(fit_residuals),
        model=model,
        summary=pd.DataFrame(rows),
        coverage_samples=pd.concat(frames, ignore_index=True),
        notes=notes,
    )
