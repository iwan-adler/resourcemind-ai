"""Sensitivity analysis: re-run every strategy while one input varies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from resourcemind.allocation.models import AllocationProblem
from resourcemind.config import AppConfig
from resourcemind.simulation.scenarios import with_demand_factor, with_resource_factor

PARAMETERS = {
    "resources": "Available teams (x current)",
    "demand": "Planning demand (x forecast)",
}


def sensitivity_sweep(problem: AllocationProblem, config: AppConfig, parameter: str, factors) -> pd.DataFrame:
    """One row per (factor, strategy) with the comparison metrics."""
    from resourcemind.pipeline import comparison_table, run_strategies

    if parameter not in PARAMETERS:
        raise ValueError(f"parameter must be one of {sorted(PARAMETERS)}")
    frames = []
    for factor in factors:
        factor = float(np.round(factor, 4))
        p = with_resource_factor(problem, factor) if parameter == "resources" else with_demand_factor(problem, factor)
        table = comparison_table(p, run_strategies(p, config))
        table.insert(0, "factor", factor)
        table.insert(1, "teams_available", int(p.available.sum()))
        table.insert(2, "planning_demand", int(p.demand.sum()))
        frames.append(table)
    return pd.concat(frames, ignore_index=True)
