"""Precomputed forecast evaluations for the default dataset.

Evaluating the forecasting models takes a few seconds (mostly fitting gradient
boosting), and the result is the same for every visitor using the default seed.
``scripts/precompute.py`` stores it in ``data/precomputed/``. Each entry carries a
fingerprint of the history, forecast settings, seed, horizon and library versions;
it is only used when the fingerprint matches, otherwise the caller recomputes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from resourcemind.config import PROJECT_ROOT, ForecastConfig
from resourcemind.forecasting.evaluation import ForecastEvaluation
from resourcemind.forecasting.models import MODEL_REGISTRY

PRECOMPUTED_DIR = PROJECT_ROOT / "data" / "precomputed"
FORMAT_VERSION = 1


def precomputed_path(seed: int, directory: Path = PRECOMPUTED_DIR) -> Path:
    return directory / f"forecast_eval_seed{seed}.pkl"


def fingerprint(demand_wide: pd.DataFrame, horizon: int, config: ForecastConfig, seed: int) -> str:
    """Hash of everything the evaluation depends on."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(demand_wide, index=True).to_numpy().tobytes())
    h.update(json.dumps(list(demand_wide.columns)).encode())
    meta = {
        "format": FORMAT_VERSION,
        "horizon": horizon,
        "seed": seed,
        "forecast": asdict(config),
        "models": sorted(MODEL_REGISTRY),
        "versions": {pkg: version(pkg) for pkg in ("scikit-learn", "numpy", "pandas")},
    }
    h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()


def save(evaluations: dict[int, tuple[str, ForecastEvaluation]], seed: int, directory: Path = PRECOMPUTED_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = precomputed_path(seed, directory)
    with path.open("wb") as f:
        pickle.dump({"format": FORMAT_VERSION, "entries": evaluations}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(
    demand_wide: pd.DataFrame, horizon: int, config: ForecastConfig, seed: int, directory: Path = PRECOMPUTED_DIR
) -> ForecastEvaluation | None:
    """The stored evaluation if its fingerprint matches, else None."""
    path = precomputed_path(seed, directory)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            stored = pickle.load(f)
        entry = stored["entries"].get(horizon) if stored.get("format") == FORMAT_VERSION else None
    except Exception:
        return None
    if entry is None:
        return None
    stored_fp, evaluation = entry
    return evaluation if stored_fp == fingerprint(demand_wide, horizon, config, seed) else None
