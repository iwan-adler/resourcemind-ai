"""Save/load synthetic datasets as CSV. Generation is fast, so loading falls back to it."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from resourcemind.config import GENERATED_DATA_DIR, AppConfig
from resourcemind.data.generate import Dataset, generate_dataset
from resourcemind.data.validate import validate_dataset

FILES = {"zones": "zones.csv", "history": "demand_history.csv", "travel": "travel_minutes.csv"}
META_FILE = "metadata.json"


def save_dataset(dataset: Dataset, directory: Path | str = GENERATED_DATA_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset.zones.to_csv(directory / FILES["zones"], index=False)
    dataset.history.to_csv(directory / FILES["history"], index=False, date_format="%Y-%m-%d")
    dataset.travel.to_csv(directory / FILES["travel"])
    (directory / META_FILE).write_text(json.dumps(dataset.meta, indent=2), encoding="utf-8")
    return directory


def read_dataset(directory: Path | str = GENERATED_DATA_DIR) -> Dataset:
    directory = Path(directory)
    zones = pd.read_csv(directory / FILES["zones"], dtype={"zone_id": str, "zone_name": str})
    history = pd.read_csv(directory / FILES["history"], parse_dates=["date"], dtype={"zone_id": str})
    travel = pd.read_csv(directory / FILES["travel"], index_col=0)
    travel.index = travel.index.astype(str)
    travel.index.name = "origin"
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    return Dataset(zones=zones, history=history, travel=travel, meta=meta)


def load_dataset(config: AppConfig, directory: Path | str | None = None) -> Dataset:
    """Load CSVs if they exist for the same seed and shape; otherwise generate in memory.

    The returned dataset is always validated; a failed validation raises.
    """
    dataset = None
    if directory is not None:
        directory = Path(directory)
        meta_path = directory / META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            same = all(
                meta.get(k) == getattr(config, k) for k in ("seed", "n_zones", "n_days", "start_date", "total_teams")
            )
            if same:
                dataset = read_dataset(directory)
    if dataset is None:
        dataset = generate_dataset(config)
    validate_dataset(dataset).raise_if_failed()
    return dataset
