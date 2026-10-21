"""Central configuration for ResourceMind AI.

All tunable parameters live in ``configs/default_scenario.json`` and are loaded
into the dataclasses below. Nothing here requires secrets or network access.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_scenario.json"
GENERATED_DATA_DIR = PROJECT_ROOT / "data" / "generated"
RESULTS_DIR = PROJECT_ROOT / "results"

# Units used throughout the codebase (shown in the UI and docs).
DEMAND_UNIT = "service requests"
RESOURCE_UNIT = "response teams"
TRAVEL_UNIT = "minutes"


@dataclass(frozen=True)
class ForecastConfig:
    lookback_days: int = 28
    validation_days: int = 28
    test_days: int = 28
    interval_lower_quantile: float = 0.1
    interval_upper_quantile: float = 0.9


@dataclass(frozen=True)
class OptimizerConfig:
    # "lexicographic": minimise weighted unmet demand first, then travel time.
    # "weighted": minimise weighted unmet demand + lambda * team-minutes travelled.
    mode: str = "lexicographic"
    # Weighted service units the planner will trade for one team-minute of travel.
    travel_penalty_lambda: float = 0.01
    time_limit_seconds: float = 10.0
    # One worker keeps CP-SAT runs deterministic for the same inputs.
    num_workers: int = 1


@dataclass(frozen=True)
class AppConfig:
    seed: int = 42
    n_zones: int = 10
    n_days: int = 240
    start_date: str = "2025-01-06"
    total_teams: int = 32
    horizon_days: int = 7
    service_rate_per_team_per_day: int = 10
    max_travel_minutes: int = 45
    min_teams_retained: int = 1
    planning_basis: str = "point"  # "point" forecast or "upper" empirical interval
    # "single": one repositioning decision for the whole horizon (default).
    # "daily": day-by-day plan; travel time eats into the moving team's shift.
    planning_mode: str = "single"
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **kwargs: Any) -> AppConfig:
        return replace(self, **kwargs)

    def validate(self) -> None:
        if not 8 <= self.n_zones <= 12:
            raise ValueError("n_zones must be between 8 and 12")
        if self.n_days < 180:
            raise ValueError("n_days must be at least 180")
        if not 1 <= self.horizon_days <= 7:
            raise ValueError("horizon_days must be between 1 and 7")
        if self.service_rate_per_team_per_day <= 0:
            raise ValueError("service_rate_per_team_per_day must be positive")
        if self.total_teams < 0:
            raise ValueError("total_teams cannot be negative")
        if self.planning_basis not in {"point", "upper"}:
            raise ValueError("planning_basis must be 'point' or 'upper'")
        if self.planning_mode not in {"single", "daily"}:
            raise ValueError("planning_mode must be 'single' or 'daily'")
        if self.optimizer.mode not in {"lexicographic", "weighted"}:
            raise ValueError("optimizer.mode must be 'lexicographic' or 'weighted'")


def config_from_dict(raw: dict[str, Any]) -> AppConfig:
    raw = dict(raw)
    forecast = ForecastConfig(**raw.pop("forecast", {}))
    optimizer = OptimizerConfig(**raw.pop("optimizer", {}))
    config = AppConfig(**raw, forecast=forecast, optimizer=optimizer)
    config.validate()
    return config


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load the JSON config file; missing keys fall back to dataclass defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return config_from_dict(raw)
