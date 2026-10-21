# Evaluation

All numbers below were produced by `python scripts/run_evaluation.py` (default seeds 42, 7, 2024, 99, 314)
on 2026-10-20 (Python 3.14.7, Windows, x64 emulation) in about 22 s. Raw outputs:
`results/evaluation_results.json`, `results/forecast_metrics.csv`, `results/strategy_comparison.csv`,
`results/summary_by_scenario.csv`. **The data are synthetic**, so these results describe behaviour of the
method on the generator, not real-world performance.

## Forecasting

### Method

* Direct multi-horizon setup: one row per (zone, origin day t, horizon h ≤ 7), target $y_{t+h}$.
  Features use only data up to t (tests check this with a tampered-future history).
* Chronological split by target date: train → 28 validation days → final 28 test days.
* Models are fitted on train and scored on validation. The **default model is the one with the lowest
  validation MAE**. Models are then refitted on train + validation and scored once on test. Test results are
  reported and are never used for any choice.
* Intervals: 10th/90th percentiles of validation **relative** residuals $(y-\hat y)/\max(\hat y, 1)$,
  separately for daily values and for the horizon total. They are empirical, not guaranteed probabilistic
  intervals; the table reports their actual test coverage (nominal 80%).

### Models

| Name | Description |
|---|---|
| `seasonal_naive` | Same weekday last week |
| `seasonal_average_3w` | Mean of the same weekday over the last 3 weeks |
| `moving_average_28d` | Trailing 28-day mean |
| `gradient_boosting` | scikit-learn HistGradientBoosting, Poisson loss, global model over zones and horizons; features: lags, 7/28-day rolling mean/std, weekday, annual sin/cos, horizon, zone (categorical) |

### Metrics

* MAE = mean |ŷ − y| (requests/day). RMSE = √mean (ŷ − y)².
* WAPE = Σ|ŷ − y| / Σ y; **undefined (NaN) when Σ y = 0** rather than reported as 0.
* Bias = mean (ŷ − y).

### Results (held-out test, horizons 1–7, mean over 5 seeds)

| Model | MAE | RMSE | WAPE | Interval coverage (nominal 80%) |
|---|---|---|---|---|
| gradient_boosting | 4.653 | 6.630 | 19.6% | 75.9% |
| moving_average_28d | 4.687 | 6.709 | 19.7% | 78.4% |
| seasonal_average_3w | 5.011 | 7.092 | 21.1% | 81.4% |
| seasonal_naive | 6.226 | 8.618 | 26.1% | 79.4% |

Model selected on validation: `moving_average_28d` in 4 of 5 seeds, `gradient_boosting` in 1.

**Interpretation.** Gradient boosting and the 28-day moving average are statistically indistinguishable
on this synthetic data (difference 0.03 requests/day, and the ranking flips between seeds; e.g. seed 42:
GB 5.80 vs MA 6.09; seed 2024: MA 3.69 vs GB 3.87). Both clearly beat the seasonal naive baseline (~25%
lower MAE). We do **not** claim the more complex model is better. Weekly seasonality in the generator is
mild relative to Poisson noise, which favours smoothing. Intervals slightly under-cover (76–81% vs 80%).

## Allocation

### Method

For each seed and case: generate data → forecast (auto-selected model) → planning demand → build one
`AllocationProblem` → run the current, greedy and optimized strategies → score all three with the same
`compute_outcome` → validate all three with the same independent validator.

Cases: the four presets plus `shortage_floor30` (the shortage preset with a 30% per-zone coverage floor).

### Metrics

Total and weighted unmet demand (requests), coverage = served / demand (bounded to [0, 1], 1 when
demand = 0), travel (team-minutes), teams moved, hard-constraint violations, runtime, worst-zone coverage
(fairness).

### Results (mean over 5 seeds)

| Case | Strategy | Unmet | Weighted unmet | Coverage | Travel (team-min) | Moved | Violations | Worst zone |
|---|---|---|---|---|---|---|---|---|
| normal | current | 80.0 | 118.5 | 96.0% | 0 | 0 | 0 | 88.3% |
| normal | greedy | 17.0 | 24.6 | 99.2% | 29.2 | 1.4 | 0 | 94.6% |
| normal | optimized | 17.0 | 24.6 | 99.2% | 29.2 | 1.4 | 0 | 94.6% |
| surge | current | 324.2 | 455.9 | 85.8% | 0 | 0 | 0 | 62.0% |
| surge | greedy | 135.2 | 183.2 | 94.5% | 112.8 | 4.0 | 0 | 78.4% |
| surge | optimized | 135.2 | 183.2 | 94.5% | 107.4 | 3.8 | 0 | 78.4% |
| shortage | current | 429.4 | 611.4 | 76.2% | 0 | 0 | 0 | 52.0% |
| shortage | greedy | 366.6 | 491.5 | 80.5% | 81.4 | 3.6 | 0 | 52.9% |
| shortage | optimized | 366.6 | 491.5 | 80.5% | 73.6 | 3.4 | 0 | 52.9% |
| disruption | current | 80.0 | 118.5 | 96.0% | 0 | 0 | 0 | 88.3% |
| disruption | greedy | 21.0 | 28.9 | 99.1% | 31.4 | 1.4 | 0 | 95.1% |
| disruption | optimized | 21.0 | 28.9 | 99.1% | 31.4 | 1.4 | 0 | 95.1% |
| shortage_floor30 | current | 429.4 | 611.4 | 76.2% | 0 | 0 | 0 | 52.0% |
| shortage_floor30 | greedy | 366.6 | 491.5 | 80.5% | 81.4 | 3.6 | 0.6 | 52.9% |
| shortage_floor30 | optimized | 366.6 | 493.7 | 80.5% | 64.0 | 3.0 | 0 | 54.4% |

The 30% floor only binds for seed 42: greedy has 3 floor violations there (mean 0.6 over 5 seeds); the
current allocation already meets the floor (its worst zone is at 35.6%).

Optimizer status: **OPTIMAL in all 25 runs**. Optimizer runtime: 22–49 ms mean per case, maximum 101 ms
(greedy ≤ 4 ms, current < 0.1 ms).

### Honest interpretation

* Rebalancing matters: both greedy and optimized cut weighted unmet demand by roughly 60–80% versus the
  legacy population-based allocation in normal/surge/disruption conditions.
* **The greedy baseline matched the optimizer's weighted unmet demand in every one of the 25 runs.**
  The problem is close to a concave resource-allocation problem, where marginal-gain greedy is often
  optimal. The optimizer's measurable advantages here are:
  1. lower travel for the same service in 7 of 25 runs (e.g. shortage seed 2024: 45 vs 77 team-minutes;
     surge seed 42: 169 vs 190);
  2. a **proof** of optimality (greedy has no guarantee);
  3. correctly handling extra policy constraints: with a 30% coverage floor (seed 42) greedy violates
     the floor in 3 zones, while the optimizer satisfies it and raises worst-zone coverage from 22.9% to 30.3%
     at the cost of +11.2 weighted unmet demand. This is the efficiency–equity trade-off made explicit.
* Fairness: minimising weighted unmet demand can *lower* the worst-off zone's coverage (shortage, seed 42:
  current 35.6% → optimized 22.9%). Priority weights alone do not prevent this.

## Robustness

Five data seeds × four presets (+ the coverage-floor case). All optimized plans passed independent
validation; the infeasible-scenario path is exercised by tests (staging capacity × 0.3, coverage floor
60% under shortage), and the app reports `INFEASIBLE` with reasons.

## Efficiency

Forecast evaluation (4 models, 3 fits each) ≈ 2–3 s per seed; end-to-end scenario run after that
≈ 0.1–0.3 s. The Streamlit app caches forecast evaluation per (seed, horizon) and each scenario run per
parameter set.

## Reproducing

```bash
python scripts/run_evaluation.py --seeds 42 7 2024 99 314
```

Same seeds and dependency versions → identical numbers (single-worker CP-SAT with a fixed seed; HGB with
fixed `random_state` and no early-stopping split). Runtimes vary with hardware.
