"""Synthetic allocation problems of any size, for scalability benchmarking.

The demo dataset is limited to 8-12 zones. This module builds larger problems
with the SAME structure as ``data/generate.py`` (density, demand model, legacy
population-based staffing, staging capacity, service caps, travel model and
45-minute repositioning limit), so solve times measured here describe the
real model at a larger scale. Everything is synthetic and seeded.
"""

from __future__ import annotations

import numpy as np

from resourcemind.allocation.models import AllocationProblem
from resourcemind.allocation.multiperiod import MultiPeriodProblem
from resourcemind.data.generate import (
    AVG_SPEED_KMH,
    DISPATCH_OVERHEAD_MIN,
    MAP_HEIGHT_KM,
    MAP_WIDTH_KM,
    ROAD_FACTOR,
    _largest_remainder,
)

BASE_ZONES = 10  # the demo map holds 10 zones on 32 x 24 km
WEEKLY_SHAPE = np.array([1.00, 0.96, 0.97, 1.00, 1.08, 1.20, 1.12])  # Mon..Sun, as in generate.py
CAPACITY_RATIO = 1.07  # total team capacity / mean demand, as in the demo dataset


def scaled_problems(
    n_zones: int,
    seed: int = 0,
    horizon: int = 7,
    rate_per_day: int = 10,
    max_travel_minutes: int = 45,
) -> tuple[AllocationProblem, MultiPeriodProblem]:
    """A single-period and a day-by-day problem for ``n_zones`` zones (same zones and teams)."""
    if n_zones < 2:
        raise ValueError("n_zones must be at least 2")
    rng = np.random.default_rng(seed)
    scale = np.sqrt(n_zones / BASE_ZONES)  # keep zones per km^2 constant as the map grows
    xy = np.column_stack(
        [rng.uniform(0, MAP_WIDTH_KM * scale, n_zones), rng.uniform(0, MAP_HEIGHT_KM * scale, n_zones)]
    )
    population = rng.integers(25_000, 140_000, size=n_zones)
    vulnerability = np.round(rng.beta(2.0, 3.0, size=n_zones), 2)
    base = population / 1000.0 * rng.uniform(0.12, 0.30, size=n_zones) * (1.0 + 0.6 * vulnerability)

    weekly = 1.0 + rng.uniform(0.6, 1.4, size=n_zones)[:, None] * (WEEKLY_SHAPE[None, :horizon] - 1.0)
    demand_daily = rng.poisson(base[:, None] * weekly).astype(int)  # (n, H)

    total_teams = max(int(round(base.sum() * CAPACITY_RATIO / rate_per_day)), n_zones)
    current = _largest_remainder(population.astype(float), total_teams, minimum=1)
    staging = np.maximum(current + 2, np.ceil(1.6 * base / rate_per_day).astype(int) + 1)
    cap_daily = np.ceil(1.7 * base).astype(int)

    dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    travel = np.round(DISPATCH_OVERHEAD_MIN + dist * ROAD_FACTOR / AVG_SPEED_KMH * 60.0).astype(int)
    np.fill_diagonal(travel, 0)
    eligible = travel <= max_travel_minutes
    np.fill_diagonal(eligible, True)

    ids = tuple(f"Z{i + 1:03d}" for i in range(n_zones))
    tiers = tuple(np.where(vulnerability >= 0.55, "high", np.where(vulnerability >= 0.35, "medium", "standard")))
    weights = np.round(1.0 + vulnerability, 2)
    retain = np.ones(n_zones, dtype=int)

    single = AllocationProblem(
        zone_ids=ids,
        zone_names=ids,
        priority_tiers=tiers,
        demand=demand_daily.sum(axis=1),
        available=current,
        staging_capacity=staging,
        service_cap=cap_daily * horizon,
        weights=weights,
        travel=travel,
        eligible=eligible,
        retain=retain,
        service_per_team=rate_per_day * horizon,
        horizon_days=horizon,
        max_travel_minutes=max_travel_minutes,
    )
    daily = MultiPeriodProblem(
        zone_ids=ids,
        zone_names=ids,
        priority_tiers=tiers,
        demand=demand_daily,
        available=current,
        staging_capacity=staging,
        service_cap=cap_daily,
        weights=weights,
        travel=travel,
        eligible=eligible,
        retain=retain,
        service_per_team=rate_per_day,
        max_travel_minutes=max_travel_minutes,
    )
    return single, daily
