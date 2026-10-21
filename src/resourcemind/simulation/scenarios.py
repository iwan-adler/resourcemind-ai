"""Scenario presets and user overrides.

Data layers are kept separate:
* historical demand  - the generated dataset (never modified in place)
* scenario history   - a copy of the history with preset changes (e.g. a recent surge)
                       that the forecasting models actually train on
* forecast demand    - model output from the scenario history
* adjusted demand    - forecast x stress multipliers

Overrides are stored separately from the dataset; ``ScenarioOverrides()`` is the
reset-to-default state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace

import numpy as np
import pandas as pd

from resourcemind.allocation.models import AllocationProblem
from resourcemind.allocation.response import response_pairs
from resourcemind.config import AppConfig
from resourcemind.data.generate import (
    AVG_SPEED_KMH,
    DISPATCH_OVERHEAD_MIN,
    MAP_HEIGHT_KM,
    MAP_WIDTH_KM,
    ROAD_FACTOR,
    Dataset,
    _largest_remainder,
)
from resourcemind.data.validate import ValidationReport, validate_history, validate_travel


@dataclass(frozen=True)
class ScenarioOverrides:
    # Changes to the scenario HISTORY (forecast models see these).
    history_surge_zones: tuple[str, ...] = ()
    history_surge_factor: float = 1.0
    history_surge_days: int = 0
    # Forward stress test applied on top of the forecast: {zone_id: multiplier}.
    demand_multipliers: dict[str, float] = field(default_factory=dict)
    # Resources and capacity.
    resource_factor: float = 1.0
    service_rate_per_day: int | None = None
    staging_capacity_factor: float = 1.0
    min_teams_retained: int | None = None
    # Equity policy: minimum share of servable demand served in every zone (0 = off).
    coverage_floor: float = 0.0
    # Priorities.
    priority_weights: dict[str, float] = field(default_factory=dict)
    # Travel and access.
    travel_multiplier: float = 1.0
    cross_river_multiplier: float = 1.0
    blocked_routes: tuple[tuple[str, str], ...] = ()
    inaccessible_zones: tuple[str, ...] = ()
    max_travel_minutes: int | None = None

    def is_default(self) -> bool:
        return self == ScenarioOverrides()

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = (
                [list(r) for r in v] if f.name == "blocked_routes" else (list(v) if isinstance(v, tuple) else v)
            )
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ScenarioOverrides:
        data = dict(data)
        for key in ("history_surge_zones", "inaccessible_zones"):
            if key in data:
                data[key] = tuple(data[key])
        if "blocked_routes" in data:
            data["blocked_routes"] = tuple(tuple(r) for r in data["blocked_routes"])
        return cls(**data)


def merge_overrides(base: ScenarioOverrides, top: ScenarioOverrides) -> ScenarioOverrides:
    """User overrides (top) layered on a preset (base). Dicts merge, sets union."""
    default = ScenarioOverrides()
    merged = {}
    for f in fields(ScenarioOverrides):
        b, t, d = getattr(base, f.name), getattr(top, f.name), getattr(default, f.name)
        if isinstance(b, dict):
            merged[f.name] = {**b, **t}
        elif f.name in ("blocked_routes", "inaccessible_zones"):
            merged[f.name] = tuple(dict.fromkeys(tuple(b) + tuple(t)))
        else:
            merged[f.name] = t if t != d else b
    return ScenarioOverrides(**merged)


@dataclass(frozen=True)
class ScenarioPreset:
    key: str
    name: str
    description: str
    overrides: ScenarioOverrides


PRESETS: dict[str, ScenarioPreset] = {
    p.key: p
    for p in [
        ScenarioPreset(
            "normal",
            "Normal operations",
            "Baseline synthetic demand. Teams are positioned by legacy population-based staffing.",
            ScenarioOverrides(),
        ),
        ScenarioPreset(
            "surge",
            "Sudden demand surge",
            "Heatwave plus a stadium event: the last 7 days of history in Eastfield, Westbrook and "
            "University Quarter run at 1.9x, and the event is assumed to continue (forecast stress x1.25).",
            ScenarioOverrides(
                history_surge_zones=("Z05", "Z06", "Z09"),
                history_surge_factor=1.9,
                history_surge_days=7,
                demand_multipliers={"Z05": 1.25, "Z06": 1.25, "Z09": 1.25},
            ),
        ),
        ScenarioPreset(
            "shortage",
            "Severe resource shortage",
            "40% of teams are unavailable (illness and vehicle maintenance), applied proportionally.",
            ScenarioOverrides(resource_factor=0.6),
        ),
        ScenarioPreset(
            "disruption",
            "Travel disruption",
            "River bridges partially closed (cross-river travel x3) and Old Town flooded: no teams can "
            "enter or leave Old Town.",
            ScenarioOverrides(cross_river_multiplier=3.0, inaccessible_zones=("Z02",)),
        ),
    ]
}


@dataclass
class ScenarioInputs:
    key: str
    name: str
    description: str
    overrides: ScenarioOverrides
    history: pd.DataFrame
    zones: pd.DataFrame  # includes teams_available, staging_capacity, priority_weight (effective)
    travel: pd.DataFrame
    eligible: pd.DataFrame
    service_rate_per_day: int
    horizon_days: int
    max_travel_minutes: int
    min_teams_retained: int
    coverage_floor: float
    changes: list[str]
    validation: ValidationReport

    @property
    def zone_ids(self) -> list[str]:
        return list(self.zones["zone_id"])

    def demand_wide(self) -> pd.DataFrame:
        wide = self.history.pivot(index="date", columns="zone_id", values="demand_total")
        return wide[self.zone_ids].sort_index()


def scale_teams(teams: np.ndarray, factor: float) -> np.ndarray:
    """Scale team counts to round(total * factor), split by largest remainder."""
    teams = np.asarray(teams, dtype=int)
    target = int(round(teams.sum() * max(factor, 0.0)))
    if teams.sum() == 0:
        return teams.copy()
    return _largest_remainder(teams.astype(float), target, minimum=0)


def apply_scenario(
    dataset: Dataset,
    config: AppConfig,
    preset_key: str = "normal",
    user_overrides: ScenarioOverrides | None = None,
    replace_preset: bool = False,
) -> ScenarioInputs:
    """Build scenario inputs from copies of the dataset. The dataset is never mutated.

    By default user overrides are layered on the preset. With ``replace_preset``
    the user overrides are the complete specification (used by the Scenario Lab,
    which starts from the preset's values and may set some back to default).
    """
    if preset_key not in PRESETS:
        raise KeyError(f"unknown scenario '{preset_key}'")
    preset = PRESETS[preset_key]
    user_overrides = user_overrides or ScenarioOverrides()
    ov = user_overrides if replace_preset else merge_overrides(preset.overrides, user_overrides)
    zones = dataset.zones.copy(deep=True)
    history = dataset.history.copy(deep=True)
    travel = dataset.travel.copy(deep=True).astype(float)
    ids = list(zones["zone_id"])
    changes: list[str] = []
    report = ValidationReport()

    def known(zone_list, label):
        bad = [z for z in zone_list if z not in ids]
        if bad:
            report.errors.append(f"{label}: unknown zones {bad}")
        return [z for z in zone_list if z in ids]

    # --- history surge (changes the data the forecaster trains on) ---
    surge_zones = known(ov.history_surge_zones, "history_surge_zones")
    if surge_zones and ov.history_surge_days > 0 and ov.history_surge_factor != 1.0:
        last = history["date"].max()
        mask = history["zone_id"].isin(surge_zones) & (
            history["date"] > last - pd.Timedelta(days=ov.history_surge_days)
        )
        for col in ["demand_medical", "demand_fire", "demand_other"]:
            history.loc[mask, col] = np.round(history.loc[mask, col] * ov.history_surge_factor).astype("int64")
        history["demand_total"] = history[["demand_medical", "demand_fire", "demand_other"]].sum(axis=1)
        changes.append(
            f"History: last {ov.history_surge_days} days x{ov.history_surge_factor:g} in {', '.join(surge_zones)}"
        )

    # --- resources ---
    zones["teams_available"] = zones["current_teams"].astype(int)
    if ov.resource_factor != 1.0:
        zones["teams_available"] = scale_teams(zones["current_teams"].to_numpy(), ov.resource_factor)
        changes.append(
            f"Resources: x{ov.resource_factor:g} -> {zones['teams_available'].sum()} of "
            f"{zones['current_teams'].sum()} teams available"
        )
    zones["staging_capacity"] = zones["staging_capacity_teams"].astype(int)
    if ov.staging_capacity_factor != 1.0:
        zones["staging_capacity"] = np.floor(
            zones["staging_capacity_teams"] * max(ov.staging_capacity_factor, 0.0)
        ).astype(int)
        changes.append(f"Staging capacity: x{ov.staging_capacity_factor:g}")
    rate = int(ov.service_rate_per_day or config.service_rate_per_team_per_day)
    if rate != config.service_rate_per_team_per_day:
        changes.append(f"Service rate: {rate} requests/team/day (default {config.service_rate_per_team_per_day})")
    if rate <= 0:
        report.errors.append("service rate must be positive")
    retain = config.min_teams_retained if ov.min_teams_retained is None else int(ov.min_teams_retained)
    if retain != config.min_teams_retained:
        changes.append(f"Minimum teams retained per zone: {retain}")

    if not 0.0 <= ov.coverage_floor <= 1.0:
        report.errors.append("coverage floor must be within [0, 1]")
    elif ov.coverage_floor > 0:
        changes.append(f"Coverage floor: every zone >= {ov.coverage_floor:.0%} of servable demand")

    # --- priorities ---
    zones["priority_weight_default"] = zones["priority_weight"]
    for z, w in ov.priority_weights.items():
        if z in ids:
            if w <= 0:
                report.errors.append(f"priority weight for {z} must be positive")
                continue
            if not math.isclose(float(w), float(zones.loc[zones.zone_id == z, "priority_weight"].iloc[0])):
                zones.loc[zones.zone_id == z, "priority_weight"] = float(w)
                changes.append(f"Priority weight {z}: {w:g}")
        else:
            report.errors.append(f"priority_weights: unknown zone {z}")

    # --- forward stress multipliers ---
    for z, m in ov.demand_multipliers.items():
        if z not in ids:
            report.errors.append(f"demand_multipliers: unknown zone {z}")
        elif m < 0:
            report.errors.append(f"demand multiplier for {z} must be non-negative")
        elif m != 1.0:
            changes.append(f"Forecast stress test {z}: x{m:g}")

    # --- travel ---
    if ov.travel_multiplier != 1.0:
        travel = travel * ov.travel_multiplier
        changes.append(f"Travel times: x{ov.travel_multiplier:g} on all routes")
    if ov.cross_river_multiplier != 1.0:
        side = zones.set_index("zone_id")["river_side"]
        cross = np.not_equal.outer(side[ids].to_numpy(), side[ids].to_numpy())
        travel = travel.where(~cross, travel * ov.cross_river_multiplier)
        changes.append(f"Cross-river travel: x{ov.cross_river_multiplier:g}")
    travel = travel.round().astype(int)
    max_travel = int(ov.max_travel_minutes or config.max_travel_minutes)
    if max_travel != config.max_travel_minutes:
        changes.append(f"Maximum repositioning time: {max_travel} min")

    eligible = pd.DataFrame(travel.to_numpy() <= max_travel, index=travel.index, columns=travel.columns)
    blocked = []
    for a, b in ov.blocked_routes:
        if a in ids and b in ids and a != b:
            eligible.loc[a, b] = False
            eligible.loc[b, a] = False
            blocked.append(f"{a}<->{b}")
        else:
            report.errors.append(f"blocked route {a}-{b} is invalid")
    if blocked:
        changes.append("Blocked routes: " + ", ".join(blocked))
    inaccessible = known(ov.inaccessible_zones, "inaccessible_zones")
    for z in inaccessible:
        eligible.loc[z, :] = False
        eligible.loc[:, z] = False
    if inaccessible:
        changes.append("Inaccessible zones: " + ", ".join(inaccessible))
    for z in ids:
        eligible.loc[z, z] = True

    report.extend(validate_history(history, ids))
    report.extend(validate_travel(travel, ids))
    zones["inaccessible"] = zones["zone_id"].isin(inaccessible)
    return ScenarioInputs(
        key=preset.key,
        name=preset.name,
        description=preset.description,
        overrides=ov,
        history=history,
        zones=zones,
        travel=travel,
        eligible=eligible,
        service_rate_per_day=rate,
        horizon_days=config.horizon_days,
        max_travel_minutes=max_travel,
        min_teams_retained=retain,
        coverage_floor=float(ov.coverage_floor),
        changes=changes,
        validation=report,
    )


def planning_demand(inputs: ScenarioInputs, forecast_totals: pd.DataFrame, basis: str = "point") -> pd.DataFrame:
    """Demand used by the allocator: forecast (point or upper interval) x stress multiplier."""
    if basis not in {"point", "upper"}:
        raise ValueError("basis must be 'point' or 'upper'")
    t = forecast_totals.set_index("zone_id").reindex(inputs.zone_ids)
    mult = pd.Series({z: inputs.overrides.demand_multipliers.get(z, 1.0) for z in inputs.zone_ids})
    column = "forecast_total" if basis == "point" else "upper_total"
    out = pd.DataFrame(
        {
            "zone_id": inputs.zone_ids,
            "forecast_total": t["forecast_total"].to_numpy(),
            "lower_total": t["lower_total"].to_numpy(),
            "upper_total": t["upper_total"].to_numpy(),
            "stress_multiplier": mult.to_numpy(),
        }
    )
    out["planning_demand"] = np.round(t[column].to_numpy() * mult.to_numpy()).astype(int)
    return out


def local_response_minutes(n_zones: int) -> float:
    """Typical response time to an incident inside a zone (synthetic, from map geometry).

    Zones share the map evenly, so each covers a square of side sqrt(area / n). The mean
    distance from a random point in a square to its centre is about 0.38 x side; the
    time uses the same dispatch overhead, road factor and speed as the travel matrix.
    """
    side_km = math.sqrt(MAP_WIDTH_KM * MAP_HEIGHT_KM / max(n_zones, 1))
    return DISPATCH_OVERHEAD_MIN + 0.38 * side_km * ROAD_FACTOR / AVG_SPEED_KMH * 60.0


def build_problem(inputs: ScenarioInputs, demand_units, response_standard: float | None = None) -> AllocationProblem:
    """Allocation problem for the scenario; ``response_standard`` (minutes) enables response-time coverage."""
    z = inputs.zones
    h = inputs.horizon_days
    pairs = None
    if response_standard is not None:
        travel = inputs.travel.to_numpy(dtype=float)
        eligible = inputs.eligible.to_numpy(dtype=bool)
        local = np.full(len(z), local_response_minutes(len(z)))
        # Blocked routes and inaccessible zones cannot be used to respond either.
        pairs = response_pairs(travel, local, response_standard, blocked=~eligible & (travel <= response_standard))
    return AllocationProblem(
        zone_ids=tuple(z["zone_id"]),
        zone_names=tuple(z["zone_name"]),
        priority_tiers=tuple(z["priority_tier"]),
        demand=np.maximum(np.asarray(demand_units, dtype=int), 0),
        available=z["teams_available"].to_numpy(dtype=int),
        staging_capacity=z["staging_capacity"].to_numpy(dtype=int),
        service_cap=(z["max_useful_service_per_day"].to_numpy(dtype=int) * h),
        weights=z["priority_weight"].to_numpy(dtype=float),
        travel=inputs.travel.to_numpy(dtype=int),
        eligible=inputs.eligible.to_numpy(dtype=bool),
        retain=np.full(len(z), inputs.min_teams_retained, dtype=int),
        service_per_team=int(inputs.service_rate_per_day * h),
        horizon_days=h,
        max_travel_minutes=inputs.max_travel_minutes,
        coverage_floor=inputs.coverage_floor,
        response_pairs=pairs,
        response_standard_minutes=response_standard,
    )


def with_resource_factor(problem, factor: float):
    """Scale available teams; works for AllocationProblem and MultiPeriodProblem."""
    return replace(problem, available=scale_teams(problem.available, factor))


def with_demand_factor(problem, factor: float):
    """Scale demand (per zone, or per zone and day); works for both problem types."""
    return replace(problem, demand=np.round(problem.demand * max(factor, 0.0)).astype(int))
