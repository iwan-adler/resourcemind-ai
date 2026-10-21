# Evaluation

All numbers below were produced by `python scripts/run_evaluation.py` (default seeds 42, 7, 2024, 99, 314)
on 2026-10-21 (Python 3.14.7, Windows, x64 emulation) in about 128 s. Raw outputs:
`results/evaluation_results.json`, `results/forecast_metrics.csv`, `results/strategy_comparison.csv`,
`results/summary_by_scenario.csv`, `results/robustness.csv`. The scalability numbers come from
`python scripts/benchmark_scalability.py` (`results/scalability.csv`). **The data are synthetic**, so these
results describe behaviour of the method on the generator, not real-world performance.

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

Cases: the four presets plus `shortage_floor30` (the shortage preset with a 30% per-zone coverage floor),
each in the default single-period mode. The same five cases are repeated in the day-by-day mode
(suffix `_daily`, see [Day-by-day planning](#day-by-day-planning) below), plus four `_daily_seasonal` cases.

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

Optimizer status: **OPTIMAL in all 25 runs**. Optimizer runtime: 43–79 ms mean per case, maximum 132 ms
(greedy ≤ 4 ms, current < 0.1 ms). The single-period numbers above are identical to those measured before
the day-by-day mode was added; only runtimes vary between runs.

### Honest interpretation

* Rebalancing matters: both greedy and optimized cut weighted unmet demand by roughly 60–80% versus the
  legacy population-based allocation in normal/surge/disruption conditions.
* **The greedy baseline matched the optimizer's weighted unmet demand in every one of the 25 runs.**
  The problem is close to a concave resource-allocation problem, where marginal-gain greedy is often
  optimal, and the 10-zone demo city is small. This is a small-problem effect: on larger synthetic
  cities the optimizer is ahead of greedy in every run from 25 zones on (see *Scalability*). The
  optimizer's measurable advantages at this size are:
  1. lower travel for the same service in 7 of 25 runs (e.g. shortage seed 2024: 45 vs 77 team-minutes;
     surge seed 42: 169 vs 190);
  2. a **proof** of optimality (greedy has no guarantee);
  3. correctly handling extra policy constraints: with a 30% coverage floor (seed 42) greedy violates
     the floor in 3 zones, while the optimizer satisfies it and raises worst-zone coverage from 22.9% to 30.3%
     at the cost of +11.2 weighted unmet demand. This is the efficiency–equity trade-off made explicit.
* Fairness: minimising weighted unmet demand can *lower* the worst-off zone's coverage (shortage, seed 42:
  current 35.6% → optimized 22.9%). Priority weights alone do not prevent this.

## Day-by-day planning

### Method

Same seeds, scenarios and forecast. `planning_mode = "daily"` builds a `MultiPeriodProblem` from the
daily forecast (rounded per day), plans every day with CP-SAT (teams may move daily; a move costs
$T_{ji}/480$ of the team's shift that day), and compares it with "keep current" and the day-by-day greedy
rule (the single-period greedy applied each day to that day's demand, without look-ahead). All three are
scored by `compute_daily_outcome` and checked by `validate_daily_plan`. See
[the model](mathematical_model.md#day-by-day-multi-period-mode).

* `<preset>_daily` and `shortage_floor30_daily`: auto-selected forecast model, as in single-period mode.
* `<preset>_daily_seasonal`: the forecast model is forced to `seasonal_average_3w`. Reason: the auto
  choice is the 28-day moving average in 4 of 5 seeds, and its forecast is the **same value for every
  day of the horizon**, so the daily problem has no day-to-day demand changes to anticipate. The
  seasonal average varies by weekday, which tests whether the optimizer gains when demand does change.
  The demand in these cases differs from the auto cases, so compare strategies within a case only.

Daily totals are not directly comparable with single-period totals: demand is rounded per day (e.g.
normal: 81.4 vs 80.0 mean unmet for "current") and moving teams lose service while travelling.

### Results (mean over 5 seeds)

| Case | Strategy | Unmet | Weighted unmet | Coverage | Travel (team-min) | Team moves | Violations | Worst zone |
|---|---|---|---|---|---|---|---|---|
| normal_daily | current | 81.4 | 120.6 | 96.0% | 0 | 0 | 0 | 88.1% |
| normal_daily | greedy | 18.4 | 26.6 | 99.2% | 29.2 | 1.4 | 0 | 94.3% |
| normal_daily | optimized | 18.4 | 26.6 | 99.2% | 29.2 | 1.4 | 0 | 94.3% |
| surge_daily | current | 325.8 | 457.9 | 85.8% | 0 | 0 | 0 | 61.7% |
| surge_daily | greedy | 142.4 | 193.2 | 94.2% | 114.8 | 4.2 | 0 | 77.6% |
| surge_daily | optimized | 141.8 | 192.4 | 94.2% | 113.0 | 4.2 | 0 | 77.6% |
| shortage_daily | current | 431.2 | 613.6 | 76.1% | 0 | 0 | 0 | 52.7% |
| shortage_daily | greedy | 370.6 | 497.2 | 80.3% | 97.6 | 3.8 | 0 | 53.7% |
| shortage_daily | optimized | 370.4 | 497.0 | 80.3% | 86.2 | 3.6 | 0 | 53.7% |
| disruption_daily | current | 81.4 | 120.6 | 96.0% | 0 | 0 | 0 | 88.1% |
| disruption_daily | greedy | 22.6 | 31.1 | 99.0% | 31.4 | 1.4 | 0 | 94.8% |
| disruption_daily | optimized | 22.6 | 31.1 | 99.0% | 31.4 | 1.4 | 0 | 94.8% |
| shortage_floor30_daily | current | 431.2 | 613.6 | 76.1% | 0 | 0 | 0 | 52.7% |
| shortage_floor30_daily | greedy | 370.6 | 497.2 | 80.3% | 97.6 | 3.8 | 4.2 | 53.7% |
| shortage_floor30_daily | optimized | 370.4 | 499.1 | 80.3% | 82.0 | 3.4 | 0 | 55.2% |
| normal_daily_seasonal | current | 100.6 | 147.8 | 94.9% | 0 | 0 | 0 | 86.5% |
| normal_daily_seasonal | greedy | 28.2 | 39.2 | 98.7% | 175.6 | 7.6 | 0 | 96.9% |
| normal_daily_seasonal | optimized | 27.4 | 38.1 | 98.7% | 153.4 | 6.8 | 0 | 96.7% |
| surge_daily_seasonal | current | 373.2 | 521.8 | 83.8% | 0 | 0 | 0 | 61.0% |
| surge_daily_seasonal | greedy | 188.2 | 256.1 | 92.1% | 302.8 | 11.0 | 0 | 76.7% |
| surge_daily_seasonal | optimized | 188.0 | 255.1 | 92.1% | 270.6 | 9.8 | 0 | 75.7% |
| shortage_daily_seasonal | current | 461.8 | 656.9 | 74.5% | 0 | 0 | 0 | 52.9% |
| shortage_daily_seasonal | greedy | 402.4 | 541.3 | 78.5% | 317.0 | 12.0 | 0 | 53.9% |
| shortage_daily_seasonal | optimized | 401.4 | 537.4 | 78.5% | 242.4 | 8.8 | 0 | 51.7% |
| disruption_daily_seasonal | current | 100.6 | 147.8 | 94.9% | 0 | 0 | 0 | 86.5% |
| disruption_daily_seasonal | greedy | 34.0 | 46.4 | 98.4% | 208.8 | 8.0 | 0 | 93.4% |
| disruption_daily_seasonal | optimized | 33.6 | 45.5 | 98.4% | 179.4 | 7.0 | 0 | 93.3% |

"Team moves" counts each team each time it moves, summed over the days.

### Optimizer vs greedy, run by run (weighted unmet demand)

`optimizer_vs_greedy` in `results/evaluation_results.json`. Gap = greedy − optimizer (positive = optimizer
better).

| Case | Runs | Optimizer better | Tie | Optimizer worse | Mean gap | Max gap |
|---|---|---|---|---|---|---|
| single-period presets (4 cases) | 20 | 0 | 20 | 0 | 0.00 | 0.00 |
| shortage_floor30 | 5 | 0 | 4 | 1 (floor binds) | −2.24 | 0.00 |
| normal_daily | 5 | 0 | 5 | 0 | 0.00 | 0.00 |
| surge_daily | 5 | 2 | 3 | 0 | 0.86 | 2.85 |
| shortage_daily | 5 | 1 | 4 | 0 | 0.15 | 0.77 |
| disruption_daily | 5 | 0 | 5 | 0 | 0.00 | 0.00 |
| shortage_floor30_daily | 5 | 0 | 4 | 1 (floor binds) | −1.95 | 0.00 |
| normal_daily_seasonal | 5 | 2 | 3 | 0 | 1.06 | 2.94 |
| surge_daily_seasonal | 5 | 2 | 3 | 0 | 1.03 | 4.29 |
| shortage_daily_seasonal | 5 | **5** | 0 | 0 | 3.88 | 6.96 |
| disruption_daily_seasonal | 5 | 2 | 3 | 0 | 0.81 | 2.37 |

"Optimizer worse" only occurs where the coverage floor binds (seed 42): greedy ignores the floor and
violates it (21 day-zone violations in daily mode), the optimizer satisfies it.

Optimizer status in daily mode: OPTIMAL in all 45 daily runs. (Before the optimizer was warm-started from
the greedy plan, one run, normal_daily_seasonal seed 42, only reached FEASIBLE in the travel stage; with
the warm start it is proven optimal with the same weighted unmet demand and 359 instead of 630
team-minutes of travel. No other result changed.) Runtime per daily optimizer run: 0.2–1.1 s mean per
default-forecast case (maximum 3.6 s), 0.7–2.6 s mean per seasonal case (maximum 9.8 s). Day-by-day
greedy ≤ 9 ms.

### Honest interpretation

* **The day-by-day optimizer now beats greedy on weighted unmet demand, but only by a small margin.**
  With the auto-selected forecast it wins in 3 of 25 daily runs (0.77–2.85 weighted requests; 0.06–2.0%
  of the greedy value) and ties the rest. With a weekday-varying forecast it wins in 11 of 20 runs,
  including all 5 shortage runs, by 0.30–6.96 weighted requests. Relative to greedy that is mostly
  0.4–1.5%; the largest relative gain, 17%, is in a run where greedy leaves little unmet (normal, seed 7:
  17.1 → 14.2). Every gain is under 0.4% of total demand. It is never worse except where the coverage
  floor makes greedy's plan infeasible.
* **Why the gain is small with the default forecast:** the moving-average forecast is flat across the
  horizon, so every day looks the same. Day-by-day greedy then makes all its moves on day 1, which is
  close to the single-period decision. The optimizer's remaining edge comes from accounting for the service
  lost while driving (at most ~0.94 requests per move with the 45-minute limit) and choosing cheaper
  moves.
* **When demand varies by day** (seasonal forecast), greedy re-chases each day's demand: it makes more
  moves (12.0 vs 8.8 per run in shortage) with more travel (317 vs 242 team-minutes) and still leaves more
  weighted demand unmet. The optimizer can anticipate later days, and in every seasonal case it now also
  travels less than greedy (e.g. normal: 153.4 vs 175.6 team-minutes).
* Worst-zone coverage is not guaranteed to improve: in shortage_daily_seasonal the optimizer's mean
  worst-zone coverage (51.7%) is lower than greedy's (53.9%). As in single-period mode, minimising
  weighted unmet demand is not a fairness guarantee; the per-day coverage floor makes equity explicit.
* The service-loss model is mild by construction (45-minute limit vs a 480-minute shift). A longer travel
  limit or a shorter effective shift would make movement more expensive and would likely widen the gap;
  that was not measured here.

## Response-time standard (20 minutes)

Cases `*_response20`: each preset, single-period, with demand counted as covered only when a team can
reach it within 20 minutes (own zone: 11.1-min local response; 16 neighbouring-zone routes qualify on the
normal map, fewer under the travel disruption). Mean over 5 seeds:

| Case | Strategy | Weighted unmet | Coverage | Travel (team-min) | Teams moved | Worst zone |
|---|---|---|---|---|---|---|
| normal | current | 10.5 | 99.6% | 0 | 0 | 95.7% |
| normal | greedy | 10.5 | 99.6% | 29.2 | 1.4 | 95.7% |
| normal | optimized | 4.0 | 99.9% | 4.2 | 0.2 | 97.5% |
| surge | current | 231.0 | 92.8% | 0 | 0 | 60.2% |
| surge | greedy | 163.3 | 95.0% | 112.8 | 4.0 | 71.5% |
| surge | optimized | 160.0 | 95.1% | 33.0 | 1.4 | 70.6% |
| shortage | current | 476.1 | 80.5% | 0 | 0 | 21.2% |
| shortage | greedy | 432.7 | 82.5% | 81.4 | 3.6 | 26.0% |
| shortage | optimized | 427.7 | 82.5% | 35.2 | 1.4 | 20.8% |
| disruption | current | 36.2 | 98.8% | 0 | 0 | 91.5% |
| disruption | greedy | 22.7 | 99.2% | 31.4 | 1.4 | 95.1% |
| disruption | optimized | 22.7 | 99.2% | 10.0 | 0.4 | 95.1% |

* Optimizer status: OPTIMAL in 20 of 20 runs (maximum 3.2 s).
* **The optimizer beats greedy in 5 of 20 runs** (by 4.7–32.4 weighted requests: normal seed 42: 32.4;
  shortage seeds 42 and 99: 8.1 and 16.9; surge seeds 7 and 42: 12.0 and 4.7) and ties the rest. It
  is never worse.
* It also moves far fewer teams for the same or better result (e.g. surge: 1.4 vs 4.0 teams, 33 vs 113
  team-minutes), because it knows a neighbouring zone's team can already cover part of the demand.
  Greedy reasons in-zone, so part of this advantage is that greedy does not see cross-zone coverage.
* Worst-zone coverage is again not guaranteed: in the shortage case the optimizer's worst zone (20.8%) is
  lower than greedy's (26.0%).
* The numbers are not comparable one-to-one with the in-zone results above: the rule for what counts as
  covered is different (stricter for distant zones, but neighbours can help).

## Robustness to forecast error

`results/robustness.csv`: every single-period plan is scored against 200 demand outcomes resampled from
the forecast model's real errors on the **held-out test** window (whole forecast windows, so errors stay
correlated across zones). The expected-outcome plan is built only from **validation** errors. Mean over
5 seeds:

| Case | Plan | Mean coverage | Bad week (5th pct) | P(coverage < 90%) | Mean weighted unmet |
|---|---|---|---|---|---|
| normal | current | 95.0% | 92.8% | 21.4% | 152.5 |
| normal | greedy / optimized (point) | 98.2% | 96.7% | 7.1% | 58.9 |
| normal | optimized (upper, cautious) | 98.3% | 97.0% | 5.8% | 55.9 |
| normal | optimized (expected outcome) | 98.1% | 96.9% | 7.1% | 63.2 |
| surge | current | 85.0% | 82.3% | 63.9% | 494.0 |
| surge | greedy / optimized (point) | 92.7% | 90.1% | 20.0% | 250.0 |
| surge | optimized (upper, cautious) | 92.3% | 89.6% | 20.0% | 259.8 |
| surge | optimized (expected outcome) | 92.8% | 90.1% | 20.0% | 248.4 |
| shortage | current | 76.5% | 72.9% | 80.0% | 619.1 |
| shortage | greedy / optimized (point) | 79.8% | 75.9% | 69.7% | 518.5 |
| shortage | optimized (upper, cautious) | 77.1% | 74.6% | 80.0% | 561.0 |
| shortage | optimized (expected outcome) | 79.8% | 75.9% | 69.7% | 516.5 |
| disruption | current | 95.0% | 92.8% | 21.4% | 152.5 |
| disruption | greedy | 98.1% | 96.6% | 4.6% | 60.5 |
| disruption | optimized (point) | 98.1% | 96.5% | 4.6% | 60.5 |
| disruption | optimized (upper, cautious) | 98.0% | 97.0% | 5.5% | 63.8 |
| disruption | optimized (expected outcome) | 98.0% | 96.9% | 9.3% | 67.1 |

* **Rebalancing is robust:** under real forecast errors the optimized plan still cuts expected weighted
  unmet demand by 49–61% versus the current allocation in normal, surge and disruption (16% in shortage,
  where teams are simply too few), and cuts the chance of a sub-90% week from 21% to 7% in normal
  operations.
* **No single "safer" plan wins everywhere.** Planning against the upper forecast helps slightly in normal
  operations (5.8% vs 7.1% chance of a sub-90% week) but clearly hurts in the shortage case (77.1% vs
  79.8% mean coverage): inflating every zone's forecast reshuffles teams towards high-priority zones
  rather than buying robustness.
* **The expected-outcome plan is about the same as the point plan:** −2.0 weighted unmet in shortage,
  −1.6 in surge, +4.3 in normal, +6.7 in disruption. Building it from validation errors does not carry
  over reliably to the test period, and it takes up to ~20 s, so it is optional in the app.

## Scalability

`results/scalability.csv`: synthetic problems with the same structure as the demo data at 10, 25, 50 and
100 zones (`simulation/scaled.py`), 3 seeds each, one CP-SAT worker, 20 s limit per optimizer stage.
Gain = greedy − optimizer weighted unmet (positive = optimizer better).

| Mode | Zones | Teams (mean) | Optimizer time (mean / max) | Proven optimal | Gain vs greedy (min–max) | Max gap to bound |
|---|---|---|---|---|---|---|
| single | 10 | 20 | 0.05 / 0.05 s | 3 of 3 | 0.0–1.5 | 0% |
| single | 25 | 57 | 0.25 / 0.31 s | 3 of 3 | 5.6–34.9 | 0% |
| single | 50 | 117 | 0.39 / 0.45 s | 3 of 3 | 22.3–114.5 | 0% |
| single | 100 | 226 | 0.98 / 1.21 s | 3 of 3 | 33.1–141.1 | 0% |
| daily | 10 | 20 | 2.3 / 3.1 s | 3 of 3 | 3.4–4.6 | 0% |
| daily | 25 | 57 | 36.0 / 40.2 s | 0 of 3 | 18.1–26.8 | 3.0% |
| daily | 50 | 117 | 40.4 / 40.4 s | 0 of 3 | 12.0–66.4 | 5.6% |
| daily | 100 | 226 | 40.7 / 40.8 s | 0 of 3 | 0.0–0.0 | 99.3% |

* **Single-period planning scales comfortably** and stays proven optimal at 100 zones in about 1 s.
* **The greedy tie is a small-problem effect.** At 10 zones greedy is within 1.5 weighted requests of
  optimal; from 25 zones on the optimizer is ahead in every run, by up to 141 weighted requests at 100
  zones.
* **Day-by-day planning is the limit.** From 25 zones it does not prove optimality within the limit (gap to
  the proven bound up to 5.6% at 50 zones). At 100 zones the solver does not improve on the greedy warm
  start within the limit, so the greedy plan is kept (gain 0, and the loose bound makes the gap 99%).
  Before the warm-start guarantee was added, the day-by-day optimizer returned plans about 700–2,500
  weighted requests *worse* than greedy at this size (with and without a plain solver hint); this
  benchmark found that problem and it is now fixed.
* Runtimes depend on hardware; parallel CP-SAT workers would be faster but are not deterministic.

## Scenario coverage

Five data seeds × four presets (+ the coverage-floor case), in both planning modes, plus the four
seasonal day-by-day cases and the four response-standard cases. All optimized plans passed independent
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
