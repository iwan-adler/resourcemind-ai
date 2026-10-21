"""Regenerate the synthetic dataset as CSV files.

Usage:  python scripts/generate_data.py [--seed 42] [--out data/generated]
"""

import argparse

import _path  # noqa: F401

from resourcemind.config import GENERATED_DATA_DIR, load_config
from resourcemind.data.generate import generate_dataset
from resourcemind.data.load import save_dataset
from resourcemind.data.validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=str(GENERATED_DATA_DIR))
    args = parser.parse_args()
    config = load_config()
    if args.seed is not None:
        config = config.with_updates(seed=args.seed)
    dataset = generate_dataset(config)
    report = validate_dataset(dataset)
    report.raise_if_failed()
    out = save_dataset(dataset, args.out)
    print(f"Synthetic dataset (seed={config.seed}) written to {out}")
    print(f"  zones: {len(dataset.zones)}, history rows: {len(dataset.history)}, warnings: {report.warnings}")


if __name__ == "__main__":
    main()
