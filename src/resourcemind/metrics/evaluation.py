"""Metric definitions shared by forecasting, allocation and fairness reporting.

Zero-denominator conventions (documented in docs/evaluation.md):
* WAPE is NaN when total actual demand is zero (undefined, not 0%).
* Coverage is 1.0 when demand is zero (nothing was left unserved).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted) -> dict[str, float]:
    """MAE, RMSE, WAPE and mean bias (predicted - actual) over paired values."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError("actual and predicted must have the same shape")
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    if a.size == 0:
        return {"mae": math.nan, "rmse": math.nan, "wape": math.nan, "bias": math.nan, "n": 0}
    err = p - a
    denom = np.abs(a).sum()
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.abs(err).sum() / denom) if denom > 0 else math.nan,
        "bias": float(np.mean(err)),
        "n": int(a.size),
    }


def coverage_ratio(served: float, demand: float) -> float:
    """Share of demand served, bounded to [0, 1]. Zero demand -> 1.0 by convention."""
    if demand <= 0:
        return 1.0
    return float(min(max(served / demand, 0.0), 1.0))


def gini(values) -> float:
    """Gini coefficient of non-negative values (0 = perfectly equal). Empty/all-zero -> 0."""
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() <= 0:
        return 0.0
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def zone_outcome_table(zone_ids, zone_names, tiers, weights, demand, served) -> pd.DataFrame:
    demand = np.asarray(demand, dtype=float)
    served = np.asarray(served, dtype=float)
    unmet = demand - served
    return pd.DataFrame(
        {
            "zone_id": list(zone_ids),
            "zone_name": list(zone_names),
            "priority_tier": list(tiers),
            "priority_weight": np.asarray(weights, dtype=float),
            "demand": demand,
            "served": served,
            "unmet": unmet,
            "weighted_unmet": unmet * np.asarray(weights, dtype=float),
            "coverage": [coverage_ratio(s, d) for s, d in zip(served, demand, strict=True)],
        }
    )


def fairness_summary(zone_table: pd.DataFrame) -> dict[str, float]:
    """Coverage dispersion across zones and priority tiers.

    These are descriptive measurements of trade-offs, not a fairness guarantee.
    """
    if zone_table.empty:
        return {"min_zone_coverage": 1.0, "max_zone_coverage": 1.0, "coverage_range": 0.0, "unmet_gini": 0.0}
    cov = zone_table["coverage"]
    out = {
        "min_zone_coverage": float(cov.min()),
        "max_zone_coverage": float(cov.max()),
        "coverage_range": float(cov.max() - cov.min()),
        "unmet_gini": gini(zone_table["unmet"].clip(lower=0)),
    }
    for tier, grp in zone_table.groupby("priority_tier"):
        out[f"coverage_tier_{tier}"] = coverage_ratio(grp["served"].sum(), grp["demand"].sum())
    return out


def tier_table(zone_table: pd.DataFrame) -> pd.DataFrame:
    if zone_table.empty:
        return pd.DataFrame(columns=["priority_tier", "zones", "demand", "served", "unmet", "coverage"])
    grp = zone_table.groupby("priority_tier", as_index=False).agg(
        zones=("zone_id", "count"),
        demand=("demand", "sum"),
        served=("served", "sum"),
        unmet=("unmet", "sum"),
    )
    grp["coverage"] = [coverage_ratio(s, d) for s, d in zip(grp["served"], grp["demand"], strict=True)]
    return grp
