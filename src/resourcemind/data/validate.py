"""Schema and consistency checks for datasets and scenario inputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ZONE_COLUMNS = {
    "zone_id": "object",
    "zone_name": "object",
    "x_km": "float",
    "y_km": "float",
    "latitude": "float",
    "longitude": "float",
    "population": "int",
    "vulnerability_index": "float",
    "priority_weight": "float",
    "base_daily_demand": "float",
    "current_teams": "int",
    "staging_capacity_teams": "int",
    "max_useful_service_per_day": "int",
}
HISTORY_COLUMNS = ["date", "zone_id", "demand_medical", "demand_fire", "demand_other", "demand_total"]


class DataValidationError(ValueError):
    """Raised when a dataset fails validation."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataValidationError("; ".join(self.errors))


def _is_kind(series: pd.Series, kind: str) -> bool:
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    if kind == "float":
        return pd.api.types.is_numeric_dtype(series)
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def validate_zones(zones: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    missing = [c for c in ZONE_COLUMNS if c not in zones.columns]
    if missing:
        report.errors.append(f"zones: missing columns {missing}")
        return report
    for col, kind in ZONE_COLUMNS.items():
        if not _is_kind(zones[col], kind):
            report.errors.append(f"zones: column '{col}' should be {kind}, got {zones[col].dtype}")
    if zones[list(ZONE_COLUMNS)].isna().any().any():
        report.errors.append("zones: missing values present")
    if zones["zone_id"].duplicated().any():
        report.errors.append("zones: duplicate zone_id values")
    for col in [
        "population",
        "base_daily_demand",
        "current_teams",
        "staging_capacity_teams",
        "max_useful_service_per_day",
    ]:
        if (zones[col] < 0).any():
            report.errors.append(f"zones: negative values in '{col}'")
    if ((zones["vulnerability_index"] < 0) | (zones["vulnerability_index"] > 1)).any():
        report.errors.append("zones: vulnerability_index must be within [0, 1]")
    if (zones["priority_weight"] <= 0).any():
        report.errors.append("zones: priority_weight must be positive")
    over = zones["current_teams"] > zones["staging_capacity_teams"]
    if over.any():
        report.warnings.append(
            "zones: current teams exceed staging capacity in " + ", ".join(zones.loc[over, "zone_id"])
        )
    return report


def validate_history(history: pd.DataFrame, zone_ids: list[str]) -> ValidationReport:
    report = ValidationReport()
    missing = [c for c in HISTORY_COLUMNS if c not in history.columns]
    if missing:
        report.errors.append(f"history: missing columns {missing}")
        return report
    if history[HISTORY_COLUMNS].isna().any().any():
        report.errors.append("history: missing values present")
    if not pd.api.types.is_datetime64_any_dtype(history["date"]):
        report.errors.append("history: 'date' must be a datetime column")
    demand_cols = [c for c in HISTORY_COLUMNS if c.startswith("demand_")]
    if (history[demand_cols] < 0).any().any():
        report.errors.append("history: negative demand values")
    parts = history[["demand_medical", "demand_fire", "demand_other"]].sum(axis=1)
    if not np.allclose(parts.to_numpy(), history["demand_total"].to_numpy()):
        report.errors.append("history: demand_total does not equal the sum of categories")
    if history.duplicated(["date", "zone_id"]).any():
        report.errors.append("history: duplicate (date, zone_id) keys")
    unknown = set(history["zone_id"]) - set(zone_ids)
    if unknown:
        report.errors.append(f"history: unknown zone_ids {sorted(unknown)}")
    absent = set(zone_ids) - set(history["zone_id"])
    if absent:
        report.errors.append(f"history: zones without history {sorted(absent)}")
    counts = history.groupby("zone_id")["date"].nunique()
    if counts.nunique() > 1:
        report.errors.append("history: zones have different numbers of observations")
    if not counts.empty and counts.min() < 180:
        report.warnings.append(f"history: only {counts.min()} daily observations per zone (< 180)")
    return report


def validate_travel(travel: pd.DataFrame, zone_ids: list[str]) -> ValidationReport:
    report = ValidationReport()
    if list(travel.index) != zone_ids or list(travel.columns) != zone_ids:
        report.errors.append("travel: matrix index/columns must match zone_ids in order")
        return report
    values = travel.to_numpy(dtype=float)
    if np.isnan(values).any():
        report.errors.append("travel: missing values present")
    if (values < 0).any():
        report.errors.append("travel: negative travel times")
    if not np.all(np.diag(values) == 0):
        report.errors.append("travel: diagonal (stay in place) must be zero")
    return report


def validate_dataset(dataset) -> ValidationReport:
    """Validate zones, history and travel together. Returns a report (does not raise)."""
    report = validate_zones(dataset.zones)
    if report.ok:
        zone_ids = list(dataset.zones["zone_id"])
        report.extend(validate_history(dataset.history, zone_ids))
        report.extend(validate_travel(dataset.travel, zone_ids))
    return report
