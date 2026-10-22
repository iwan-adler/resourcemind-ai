"""Precompute forecast evaluations for the default seed (all horizons 1-7).

Usage:  python scripts/precompute.py [--seed 42]

Run again after changing the data generator, forecasting code, forecast settings
or library versions; the app falls back to computing on the fly when the stored
fingerprint no longer matches.
"""

import argparse

import _path  # noqa: F401

from resourcemind.config import load_config
from resourcemind.data.generate import generate_dataset
from resourcemind.forecasting import precomputed
from resourcemind.forecasting.evaluation import evaluate_forecasters
from resourcemind.forecasting.features import MAX_HORIZON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config()
    seed = config.seed if args.seed is None else args.seed
    wide = generate_dataset(config.with_updates(seed=seed)).demand_wide()
    entries = {}
    for horizon in range(1, MAX_HORIZON + 1):
        fp = precomputed.fingerprint(wide, horizon, config.forecast, seed)
        entries[horizon] = (fp, evaluate_forecasters(wide, horizon, config.forecast, random_state=seed))
        print(f"horizon {horizon}: done")
    path = precomputed.save(entries, seed)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
