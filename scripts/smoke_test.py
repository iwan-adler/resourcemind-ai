"""Fast health check: run the full pipeline once and verify core invariants.

Usage:  python scripts/smoke_test.py   (exit code 0 = healthy)
"""

import sys

import _path  # noqa: F401

from resourcemind.config import load_config
from resourcemind.pipeline import PipelineSettings, run_pipeline


def main() -> int:
    result = run_pipeline(load_config(), PipelineSettings())
    opt = result.runs["optimized"]
    current = result.runs["current"]
    checks = {
        "optimizer returned OPTIMAL or FEASIBLE": opt.result.status in ("OPTIMAL", "FEASIBLE"),
        "optimized plan passes independent validation": not opt.violations,
        "forecast covers every zone": len(result.forecast.totals) == result.problem.n,
        "optimized weighted unmet <= current allocation": opt.outcome is not None
        and opt.outcome.weighted_unmet <= current.outcome.weighted_unmet + 1e-9,
    }
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"total pipeline time: {result.timings['total']:.2f}s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
