"""Deterministic synthetic data generation.

Everything produced here is SYNTHETIC. Zone names, coordinates, populations and
demand are invented for demonstration and do not describe any real place.

Demand model (per zone i, day t, category c):

    lambda_it = base_i * weekly_i[dow(t)] * annual_i(t) * trend_i(t) * spike_it * citywide_t
    demand_itc ~ Poisson(lambda_it * share_ic)

Spikes are rare zone-level multipliers (local incidents); city-wide events
multiply every zone on the same day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from resourcemind.config import AppConfig

ZONE_NAMES = [
    "Harbor District",
    "Old Town",
    "Northgate",
    "Riverside",
    "Eastfield",
    "Westbrook",
    "Hillcrest",
    "Industrial Park",
    "University Quarter",
    "Southport",
    "Lakeside",
    "Airport Fringe",
]
CATEGORIES = ["medical", "fire", "other"]

# Schematic map extent (km). The "river" splits the map at RIVER_X_KM; the
# travel-disruption scenario closes bridges across it.
MAP_WIDTH_KM = 32.0
MAP_HEIGHT_KM = 24.0
RIVER_X_KM = 16.0
# Synthetic degrees: km / 111 on a local plane at (0, 0). Not a real location.
KM_PER_DEGREE = 111.0

ROAD_FACTOR = 1.25  # road distance / straight-line distance
AVG_SPEED_KMH = 35.0
DISPATCH_OVERHEAD_MIN = 4.0


@dataclass
class Dataset:
    """Container for one synthetic dataset. Treat as read-only."""

    zones: pd.DataFrame
    history: pd.DataFrame
    travel: pd.DataFrame  # square matrix, index = origin zone_id, columns = destination
    meta: dict = field(default_factory=dict)

    @property
    def zone_ids(self) -> list[str]:
        return list(self.zones["zone_id"])

    def demand_wide(self) -> pd.DataFrame:
        """Daily total demand as a (date x zone_id) matrix."""
        wide = self.history.pivot(index="date", columns="zone_id", values="demand_total")
        return wide[self.zone_ids].sort_index()

    def copy(self) -> Dataset:
        return Dataset(
            zones=self.zones.copy(deep=True),
            history=self.history.copy(deep=True),
            travel=self.travel.copy(deep=True),
            meta=dict(self.meta),
        )


def _largest_remainder(weights: np.ndarray, total: int, minimum: int = 0) -> np.ndarray:
    """Split an integer total proportionally to weights (deterministic)."""
    n = len(weights)
    if total <= 0 or n == 0:
        return np.zeros(n, dtype=int)
    minimum = min(minimum, total // n)
    remaining = total - minimum * n
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    raw = w * remaining
    base = np.floor(raw).astype(int)
    leftover = remaining - base.sum()
    order = np.lexsort((np.arange(n), -(raw - base)))  # largest fraction first, stable
    base[order[:leftover]] += 1
    return base + minimum


def _place_zones(n: int, rng: np.random.Generator, min_dist_km: float = 5.0) -> np.ndarray:
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n:
        attempts += 1
        p = (rng.uniform(2.0, MAP_WIDTH_KM - 2.0), rng.uniform(2.0, MAP_HEIGHT_KM - 2.0))
        if attempts > 10_000 or all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_dist_km for q in points):
            points.append(p)
    return np.array(points)


def generate_zones(config: AppConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_zones
    xy = _place_zones(n, rng)
    population = rng.integers(25_000, 140_000, size=n)
    vulnerability = np.round(rng.beta(2.0, 3.0, size=n), 2)
    # Requests per day: population-driven, higher where the vulnerability proxy is higher.
    per_capita = rng.uniform(0.12, 0.30, size=n)
    base_demand = np.round(population / 1000.0 * per_capita * (1.0 + 0.6 * vulnerability), 2)

    # Legacy staffing: teams historically assigned in proportion to population,
    # NOT to demand. This is the "static allocation" the planner improves on.
    current_teams = _largest_remainder(population.astype(float), config.total_teams, minimum=1)

    rate = config.service_rate_per_team_per_day
    needed = np.ceil(1.6 * base_demand / rate).astype(int) + 1
    staging_capacity = np.maximum(current_teams + 2, needed)
    # Maximum useful service per day: beyond this, extra teams add nothing
    # (e.g. limited by local hospital intake or access).
    max_useful_service = np.ceil(1.7 * base_demand).astype(int)

    priority_weight = np.round(1.0 + vulnerability, 2)
    tiers = np.where(vulnerability >= 0.55, "high", np.where(vulnerability >= 0.35, "medium", "standard"))

    return pd.DataFrame(
        {
            "zone_id": [f"Z{i + 1:02d}" for i in range(n)],
            "zone_name": ZONE_NAMES[:n],
            "x_km": np.round(xy[:, 0], 2),
            "y_km": np.round(xy[:, 1], 2),
            "latitude": np.round(xy[:, 1] / KM_PER_DEGREE, 5),
            "longitude": np.round(xy[:, 0] / KM_PER_DEGREE, 5),
            "river_side": np.where(xy[:, 0] < RIVER_X_KM, "west", "east"),
            "population": population,
            "vulnerability_index": vulnerability,
            "priority_tier": tiers,
            "priority_weight": priority_weight,
            "base_daily_demand": base_demand,
            "current_teams": current_teams,
            "staging_capacity_teams": staging_capacity,
            "max_useful_service_per_day": max_useful_service,
        }
    )


def generate_travel_matrix(zones: pd.DataFrame) -> pd.DataFrame:
    xy = zones[["x_km", "y_km"]].to_numpy()
    dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    minutes = DISPATCH_OVERHEAD_MIN + dist * ROAD_FACTOR / AVG_SPEED_KMH * 60.0
    minutes = np.round(minutes).astype(int)
    np.fill_diagonal(minutes, 0)
    ids = list(zones["zone_id"])
    return pd.DataFrame(minutes, index=pd.Index(ids, name="origin"), columns=ids)


def generate_history(config: AppConfig, zones: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_days, n_zones = config.n_days, len(zones)
    dates = pd.date_range(config.start_date, periods=n_days, freq="D")
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    weekly_shape = np.array([1.00, 0.96, 0.97, 1.00, 1.08, 1.20, 1.12])  # Mon..Sun
    weekly_amp = rng.uniform(0.6, 1.4, size=n_zones)
    annual_amp = rng.uniform(0.05, 0.12, size=n_zones)
    annual_phase = rng.uniform(0, 2 * np.pi, size=n_zones)
    growth = rng.uniform(0.0, 0.08, size=n_zones)
    shares = rng.dirichlet([12.0, 2.5, 4.0], size=n_zones)

    t = np.arange(n_days) / max(n_days - 1, 1)
    citywide = np.where(rng.random(n_days) < 0.012, rng.uniform(1.3, 1.5, n_days), 1.0)

    rows = []
    for i, zone in enumerate(zones.itertuples(index=False)):
        weekly = 1.0 + weekly_amp[i] * (weekly_shape[dow] - 1.0)
        annual = 1.0 + annual_amp[i] * np.sin(2 * np.pi * doy / 365.25 + annual_phase[i])
        trend = 1.0 + growth[i] * t
        spikes = np.where(rng.random(n_days) < 0.02, rng.uniform(1.6, 2.4, n_days), 1.0)
        lam = zone.base_daily_demand * weekly * annual * trend * spikes * citywide
        counts = rng.poisson(lam[:, None] * shares[i][None, :])
        frame = pd.DataFrame(counts, columns=[f"demand_{c}" for c in CATEGORIES])
        frame.insert(0, "zone_id", zone.zone_id)
        frame.insert(0, "date", dates)
        rows.append(frame)

    history = pd.concat(rows, ignore_index=True)
    history["demand_total"] = history[[f"demand_{c}" for c in CATEGORIES]].sum(axis=1)
    for col in [f"demand_{c}" for c in CATEGORIES] + ["demand_total"]:
        history[col] = history[col].astype("int64")
    return history.sort_values(["date", "zone_id"]).reset_index(drop=True)


def generate_dataset(config: AppConfig) -> Dataset:
    """Generate the full synthetic dataset. Same config + seed -> identical output."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    zones = generate_zones(config, rng)
    travel = generate_travel_matrix(zones)
    history = generate_history(config, zones, rng)
    meta = {
        "provenance": "synthetic (generated locally by resourcemind.data.generate)",
        "seed": config.seed,
        "n_zones": config.n_zones,
        "n_days": config.n_days,
        "start_date": config.start_date,
        "total_teams": config.total_teams,
        "demand_unit": "service requests per day",
        "travel_unit": "minutes",
    }
    return Dataset(zones=zones, history=history, travel=travel, meta=meta)
