# ResourceMind AI — complete project report

*Adaptive Emergency Resource Allocation Under Changing Demand*

This document records everything built for ResourceMind AI, in one place: what the system does, how each
part works, the decisions behind it, every measured result, how it was verified, what changed along the
way, and what is still open. All numbers come from the committed evaluation outputs in `results/` or from
the running app; nothing is estimated or invented.

> **ResourceMind AI is a decision-support prototype on synthetic data.** It is not a validated emergency
> dispatch system, and its recommendations are not operational instructions.

---

## Contents

1. [Summary](#1-summary)
2. [The problem](#2-the-problem)
3. [What was built](#3-what-was-built)
4. [Environment and setup decisions](#4-environment-and-setup-decisions)
5. [Architecture](#5-architecture)
6. [Synthetic data](#6-synthetic-data)
7. [Demand forecasting](#7-demand-forecasting)
8. [Gap analysis](#8-gap-analysis)
9. [The optimization model (single-period)](#9-the-optimization-model-single-period)
10. [Baselines and independent validation](#10-baselines-and-independent-validation)
11. [Day-by-day planning (multi-period)](#11-day-by-day-planning-multi-period)
11a. [Second round: scale, robustness, explanations, response time](#11a-second-round-scale-robustness-explanations-response-time)
12. [Scenarios, Scenario Lab and sensitivity](#12-scenarios-scenario-lab-and-sensitivity)
13. [The dashboard](#13-the-dashboard)
14. [Evaluation results](#14-evaluation-results)
15. [Testing, linting and CI](#15-testing-linting-and-ci)
16. [Responsible AI](#16-responsible-ai)
17. [Problems found and fixed along the way](#17-problems-found-and-fixed-along-the-way)
18. [Repository, history and deliverables](#18-repository-history-and-deliverables)
19. [How to run everything](#19-how-to-run-everything)
20. [Deployment status](#20-deployment-status)
21. [Design decisions to defend to judges](#21-design-decisions-to-defend-to-judges)
22. [Limitations and future work](#22-limitations-and-future-work)

---

## 1. Summary

| Item | Status |
|---|---|
| Working Streamlit app (6 sections, offline, no keys) | Done |
| Deterministic synthetic data (10 zones × 240 days) with validation | Done |
| Forecasting: 3 baselines + gradient boosting, chronological evaluation, empirical intervals | Done |
| CP-SAT optimizer (lexicographic or weighted), infeasibility diagnosis | Done |
| Baselines (current, greedy) + independent validator | Done |
| Optional day-by-day (multi-period) planning mode | Done (merged in PRs #1 and #2) |
| Scalability benchmark (10–100 zones), robustness check, "what if?" explanations, response-time standard | Done (branch `feature/improvements`) |
| 4 scenario presets, Scenario Lab, coverage floor, sensitivity analysis | Done |
| Exports: CSV, JSON run record, Markdown summary | Done |
| Reproducible evaluation (5 seeds) with committed results | Done |
| 110 automated tests, ruff lint/format, GitHub Actions CI (Python 3.12 + 3.13) | Done, green |
| Documentation (README + 7 docs + this report), screenshots | Done |
| Public GitHub repository | Done — https://github.com/iwan-adler/resourcemind-ai |
| 11-slide pitch deck with speaker notes | Done (private until shared) |
| Streamlit Community Cloud deployment | **Not done — needs the owner’s sign-in** |

**Headline results (synthetic data, mean over 5 seeds):**

* Rebalancing teams lifts coverage in normal operations from 96.0% to 99.2% and cuts weighted unmet
  demand from 118.5 to 24.6.
* Forecasting: gradient boosting (test MAE 4.65 requests/day) and a 28-day moving average (4.69) are
  effectively tied; both beat seasonal naive (6.23).
* **On the 10-zone demo city, a greedy heuristic matched the optimizer’s weighted unmet demand in all 25
  single-period runs.** The optimizer still used less travel in 7 of 25 runs, proved optimality, and
  satisfied an equity floor that greedy violated. **That tie is a small-problem effect:** on synthetic
  cities of 25–100 zones the optimizer beat greedy in every run (by up to 141 weighted requests) and
  stayed proven optimal in about 1 s.
* **In day-by-day planning, the optimizer beats greedy** in 3 of 25 runs with the default forecast and
  11 of 20 runs with a weekday-varying forecast. Every gain is under 0.4% of total demand.
* **With a 20-minute response-time standard,** the optimizer beats greedy in 5 of 20 runs, never loses,
  and moves far fewer teams.
* **Under real forecast errors** the optimized plan keeps its advantage (normal: 7% vs 21% chance of a
  sub-90% week compared with today’s allocation).

---

## 2. The problem

Emergency services have a fixed number of response teams. Demand changes over time and across zones, so
a static allocation leaves some zones short while others have spare capacity. The project goal was a
tool that:

1. forecasts near-future demand per zone;
2. estimates shortages and surpluses;
3. recommends a reallocation using **constrained optimization** (not an LLM);
4. explains the plan, its constraints and trade-offs;
5. simulates alternative scenarios;
6. compares against credible baselines;
7. evaluates accuracy, coverage, efficiency, robustness, fairness and runtime.

Hard constraints from the brief: zero budget, no paid APIs or keys, no external LLM, no network needed for
the core demo, deterministic synthetic data, honest reporting (no fabricated metrics, no fake UI), and a
codebase a student can understand and defend.

---

## 3. What was built

* **`app.py`** — a Streamlit dashboard: Command overview, Demand forecasts, Allocation planner, Scenario
  lab, Strategy evaluation, Methodology. Every number on screen comes from real computation.
* **`src/resourcemind/`** — the engine, split into data, forecasting, allocation, simulation, metrics,
  visualization and UI modules, tied together by `pipeline.run_pipeline`.
* **`scripts/`** — `generate_data.py` (CSV export), `run_evaluation.py` (multi-seed evaluation →
  `results/`), `smoke_test.py` (health check).
* **`tests/`** — 110 pytest tests, including headless UI tests with Streamlit’s `AppTest`.
* **`docs/`** — architecture, mathematical model, data dictionary, evaluation, assumptions & limitations,
  demo script, this report, and screenshots.
* **`.github/workflows/tests.yml`** — CI on Ubuntu with Python 3.12 and 3.13.

---

## 4. Environment and setup decisions

* The working folder started **empty**; there was no existing work to preserve.
* The development machine is **Windows on ARM64**. Google OR-Tools publishes no Windows-ARM64 wheels, so
  installation failed on the native ARM Python 3.13. The project therefore uses the **x64 Python 3.14.7**
  (running under emulation) in a local `.venv`. A CP-SAT smoke test confirmed it works.
* CI later verified the project on **Linux with Python 3.12 and 3.13** (Streamlit Community Cloud’s
  default is 3.12, checked on 2026-10-20).
* Dependencies are **pinned exactly** in `requirements.txt` (streamlit 1.66.0, pandas 3.0.6, numpy 2.5.4,
  scikit-learn 1.9.1, ortools 9.15.6755, plotly 7.1.0); `requirements-dev.txt` adds pytest 9.1.1 and ruff
  0.17.1. A clean install into a fresh virtual environment was tested and passed.
* **statsmodels was deliberately left out.** A statistical time-series model was optional; evaluating it
  with daily rolling forecasts would need hundreds of refits per run and slow the app.

---

## 5. Architecture

```
data/generate.py ──► Dataset (zones, history, travel)  [read-only]
data/validate.py ──► schema / range / key checks
        │
simulation/scenarios.py ──► ScenarioInputs (copies + preset & user overrides, re-validated)
        │
forecasting/features.py ──► leakage-safe (zone, origin, horizon) rows
forecasting/models.py   ──► seasonal naive · seasonal average · moving average · gradient boosting
forecasting/evaluation.py ─► chronological evaluation, empirical intervals, future forecast
        │
planning demand (forecast × stress) ──► AllocationProblem / MultiPeriodProblem
        │
allocation/baselines.py   ──► current, greedy (+ daily versions)  ┐ shared scoring
allocation/optimizer.py   ──► CP-SAT single-period                ├► compute_outcome / compute_daily_outcome
allocation/multiperiod.py ──► CP-SAT day-by-day                   ┘ independent validation (validation.py)
allocation/explain.py     ──► per-zone rationale, plan summary
metrics/evaluation.py     ──► forecast metrics, coverage, fairness
simulation/sensitivity.py ──► re-solve every strategy across a parameter range
        │
pipeline.run_pipeline() ──► used by the app, the evaluation script and the tests
        │
app.py + visualization/charts.py (Plotly) + ui/components.py (styling)
```

**Principles:**

* **One pipeline.** The UI cannot show anything the tests and the evaluation do not also compute.
* **Separate data layers.** Historical demand, scenario history, forecast and stress-tested demand are
  distinct objects; overrides are applied to copies, so “reset” just discards them.
* **Shared scoring, independent validation.** Every strategy is scored by the same function and checked
  by the same validator; the solver’s own objective value is never used for reporting.
* **Deterministic.** Fixed seeds; CP-SAT with one worker; gradient boosting with no random early-stopping
  split.
* **Offline.** No map tiles, APIs, LLMs or downloads at runtime.

---

## 6. Synthetic data

Generated by `src/resourcemind/data/generate.py` from seed 42 (configurable). **Entirely synthetic:** the
zone names are invented and the coordinates are a schematic km grid that does not describe a real place.

* **10 zones** (Harbor District, Old Town, Northgate, Riverside, Eastfield, Westbrook, Hillcrest,
  Industrial Park, University Quarter, Southport) on a 32 × 24 km grid split by a schematic river.
* **240 days** (2025-01-06 to 2025-09-02) of daily demand in three categories (medical, fire, other).
* **Demand model:** Poisson counts with rate = base level × weekly pattern (weekend peak) × annual cycle
  (±5–12%) × growth trend (0–8%) × zone spikes (2% of days, ×1.6–2.4) × rare city-wide events.
* **Zone attributes:** population, a vulnerability proxy (Beta(2, 3), 0–1), priority tier, priority
  weight (`1 + vulnerability`), current teams, staging capacity and maximum useful service per day.
* **Legacy allocation:** 32 teams split **in proportion to population, not demand**. That creates the
  realistic mismatch the planner fixes.
* **Travel matrix:** minutes = 4 + road distance (1.25 × straight line) at 35 km/h; routes over 45 minutes
  are ineligible for repositioning.
* **Service rate:** one team handles 10 requests/day (70 per 7-day horizon). The first draft used 5,
  which made even “normal operations” a severe shortage (~300 requests/day vs 160 capacity). Raising it to
  10 gave total capacity ≈ 1.07 × demand but maldistributed, which is the problem being studied.
* **Validation** (`data/validate.py`): columns and types, missing values, negatives, duplicate keys,
  unknown/missing zones, equal observation counts, categories summing to totals, travel matrix shape and
  zero diagonal. Scenario inputs are re-validated on every change.

Example gaps (normal operations, 7-day horizon): Eastfield (Z05) is 113 requests short while Northgate
(Z03) has 154 requests of spare capacity; overall the current allocation covers 89.1% of demand.

---

## 7. Demand forecasting

**Setup.** A direct multi-horizon design: one supervised row per (zone, forecast day *t*, horizon *h* ≤ 7)
with target demand on day *t + h*. Features use **only data up to day *t***: last value, 7- and 28-day
rolling mean, 28-day standard deviation, the same weekday 1/2/3 weeks before the target, target weekday,
annual sine/cosine, horizon and zone. A test feeds deliberately corrupted *future* data and confirms the
forecast does not change.

**Models.**

| Name | Description |
|---|---|
| `seasonal_naive` | Same weekday last week |
| `seasonal_average_3w` | Mean of the same weekday over the last 3 weeks |
| `moving_average_28d` | Trailing 28-day mean |
| `gradient_boosting` | scikit-learn HistGradientBoosting, Poisson loss (non-negative), one global model |

**Chronological split** by target date: train → 28 validation days → final 28 test days. Models are
fitted on train and scored on validation; **the default model is the one with the lowest validation
MAE**. They are then refitted on train + validation and scored once on test. Test results are reported
and never used for any choice.

**Uncertainty.** 10th/90th percentiles of validation *relative* residuals, separately for daily values and
for the horizon total, labelled as an **empirical interval, not a guaranteed probabilistic interval**.
The app reports its actual test coverage.

**Metrics.** MAE, RMSE, WAPE (undefined, not 0%, when actual demand is zero) and bias.

---

## 8. Gap analysis

For each zone: `gap = planning demand − current teams × service per team` (in requests). Positive means
shortage, negative means spare capacity. The gap table also shows the forecast interval, stress
multiplier, staging capacity, service cap and coverage before allocation. Teams (resources) and requests
(service units) are always shown in separate, labelled units.

---

## 9. The optimization model (single-period)

Google OR-Tools **CP-SAT**, all-integer. *x[j, i]* = teams from origin *j* stationed in zone *i* for the
horizon.

**Constraints**

| Rule | Formulation |
|---|---|
| Conservation | Σᵢ x[j,i] = A_j (every team is somewhere) |
| Retention | x[j,j] ≥ min(1, A_j) (each zone keeps a team) |
| Staging capacity | Σⱼ x[j,i] ≤ M_i |
| Service | S_i ≤ D_i, S_i ≤ K_i, S_i ≤ R · X_i |
| Unmet | U_i = D_i − S_i |
| Routes | x[j,i] = 0 if travel > 45 min, route blocked or zone inaccessible |
| Equity floor (optional) | S_i ≥ ⌈f · min(D_i, K_i)⌉ |

**Objective (default, lexicographic).** (1) Minimize weighted unmet demand Σ wᵢ Uᵢ; (2) holding that
optimum, minimize team-minutes travelled. This avoids inventing an exchange rate between travel and
unserved people. A **weighted-sum** mode (λ = weighted requests per team-minute, default 0.01) is
also available and exposed in the UI.

**Solver settings.** One worker and a fixed seed (deterministic). `linearization_level = 2` was added
after one surge instance could not prove optimality in 30 s; with it, the same case is proven optimal in
0.05 s.

**Status honesty.** OPTIMAL, FEASIBLE (valid but not proven optimal), INFEASIBLE (no plan; reasons and the
smallest staging-capacity increase that would fix it), UNKNOWN (no solution in time). A plan that ever
failed independent validation would be marked INVALID and rejected.

**Explanations.** Each zone gets a rationale (why it gained, released or kept teams) and a limiting
factor: demand fully served, service cap K, staging capacity M, or team capacity.

---

## 10. Baselines and independent validation

* **Current allocation** — every team stays put.
* **Greedy** — repeatedly move one team along the eligible route with the best (weighted gain − weighted
  loss) / (1 + travel/60), until no move helps. It respects staging, retention and routes, but does
  **not** know about the equity floor.
* **Independent validator** (`allocation/validation.py`) — recomputes conservation, retention, route
  eligibility, staging capacity, integrality, non-negativity and the equity floor from the raw problem
  data, for every strategy. All strategies are compared on identical inputs.

---

## 11. Day-by-day planning (multi-period)

Built as an optional mode (sidebar: *Day by day (teams move daily)*), merged via
[PR #1](https://github.com/iwan-adler/resourcemind-ai/pull/1). The single-period mode stays the default
and its results are unchanged (all 75 single-period evaluation rows identical before and after).

**Model** (`allocation/multiperiod.py`): for each day *d*, teams stationed X[i,d] and moves y[j,i,d].

* Flow conservation: X[i,d] = X[i,d−1] + inflow − outflow, starting from current positions.
* Teams leave only from where they started the day (at most one hop per day); retention and staging
  capacity hold every day.
* **Travel consumes service:** a team moving j → i on day d spends T_ji of a 480-minute shift driving, so
  it serves R × (1 − T_ji/480) requests that day (an assumption, documented).
* Same lexicographic objective, CP-SAT settings and status semantics.
* Its own independent validator (`validate_daily_plan`) and baselines: keep current every day, and
  day-by-day greedy (the single-period greedy rule applied one day at a time, without look-ahead).

**Review before merge.** The model logic, validator and baselines were read line by line. The full test
suite (81) and smoke test passed. The evaluation was re-run from scratch and reproduced the committed
results exactly. Two summary percentages in its write-up were misstated and were corrected (see §17).

**Every page follows the mode** ([PR #2](https://github.com/iwan-adler/resourcemind-ai/pull/2)). In daily
mode the Command overview’s cards, status pills and plan summary use the daily plan; its map has a day
slider showing that day’s moves and end-of-day positions; the Scenario Lab compares day-by-day plans; and
the sensitivity analysis re-solves the day-by-day strategies. Reviewed (the per-day map matrix is rebuilt
from each day’s starting positions, so it matches the stored plan; single-period paths unchanged) and
merged after CI passed; 84 tests pass on the merged `main`.

---

## 11a. Second round: scale, robustness, explanations, response time

Built on branch `feature/improvements`. All four are optional or additive: with them switched off, every
earlier evaluation result is reproduced exactly, except one day-by-day run that improved (below).

**Scalability benchmark** (`scripts/benchmark_scalability.py`, `simulation/scaled.py`). Synthetic cities of
10, 25, 50 and 100 zones with the same structure as the demo data, 3 seeds each.

* Single-period planning: proven optimal at every size, at most 1.2 s at 100 zones (226 teams). The
  optimizer is ahead of greedy in every run from 25 zones on: by 5.6–34.9 weighted requests at 25 zones,
  22.3–114.5 at 50 and 33.1–141.1 at 100. The 10-zone tie was a small-problem effect.
* Day-by-day planning: proven optimal at 10 zones (≤ 3.1 s); from 25 zones it stops at the 20 s-per-stage
  limit without proof (at most 3.0% from the proven bound at 25 zones, 5.6% at 50). The first benchmark run
  showed it returning plans 700–2,500 weighted requests *worse* than greedy at 100 zones. Fix: warm-start
  from the greedy plan, and keep that plan (when it passes the validator) if the solver cannot beat it in
  time. It is now never worse than greedy; at 100 zones it keeps the greedy plan. The warm start also made
  the one previously FEASIBLE evaluation run (normal, weekday-varying forecast, seed 42) proven OPTIMAL,
  with the same weighted unmet and 359 instead of 630 team-minutes of travel.
* Both optimizers now record CP-SAT’s proven lower bound and the optimality gap.

**Robustness to forecast error** (`simulation/robustness.py`, Strategy evaluation page). Each plan is
scored against 200 demand outcomes built by resampling whole forecast windows of the model’s real errors
from the held-out **test** period. Optimized plans keep their advantage: in normal operations a 7% chance
of a sub-90% week versus 21% for today’s allocation, and 49–61% less expected weighted unmet demand outside
the shortage case. Planning against the upper forecast did not reliably help (worse in shortage), and a
sample-average “expected outcome” plan built from validation errors was about the same as the point plan
(−2.0 to +6.7 weighted unmet). An early version of this test drew both plan-building and scoring errors from
the same windows; that was caught and corrected before reporting.

**“What if?” explanations** (`allocation/counterfactual.py`, Allocation planner → What if?). Re-solves the
real model with one move forced or forbidden and reports the exact change. Example (normal, seed 42):
forbidding either recommended move leaves an equally good plan (+4 and +32 team-minutes of travel), and
forcing a team from Harbor District to Old Town costs 21.5 weighted requests. Impossible moves are refused
with the reason (e.g. a 57-minute trip over the 45-minute limit).

**Response-time standard** (`allocation/response.py`, sidebar → Planning settings). Demand counts as covered
only when a team can reach it within the standard; teams in neighbouring zones can help, sharing their
capacity. Local response time is 11.1 min on the demo map (from its geometry). With 20 minutes the
optimizer is OPTIMAL in 20 of 20 runs (≤ 3.2 s), beats greedy in 5 of 20 (by 4.7–32.4 weighted requests),
never loses, and moves far fewer teams (surge: 1.4 vs 4.0 teams, 33 vs 113 team-minutes). Greedy still
reasons in-zone, which accounts for part of that advantage. Example: in normal operations the plan becomes
a single move (Eastfield → Westbrook) and weighted unmet falls from 52.4 to 20.0. Single-period only.

---

## 12. Scenarios, Scenario Lab and sensitivity

**Presets** (each changes real inputs, not just labels):

| Preset | What changes |
|---|---|
| Normal operations | Nothing; legacy population-based staffing |
| Sudden demand surge | Last 7 days of history × 1.9 in Eastfield, Westbrook, University Quarter (so the forecast genuinely rises), plus an explicit × 1.25 forward stress |
| Severe resource shortage | 40% of teams unavailable (32 → 19), applied proportionally |
| Travel disruption | Cross-river travel × 3 (bridges) and Old Town inaccessible (flooding) |

**Scenario Lab** (applies on top of the chosen preset, never modifying the dataset; reset restores the
preset): per-zone demand stress, priority weights and accessibility; available teams; service rate;
staging capacity; minimum teams per zone; **coverage floor**; travel multipliers; maximum repositioning
time; blocked routes. After **Run simulation** the app re-validates, re-forecasts if history changed,
re-runs all three strategies, re-validates every plan, and shows what changed against the preset.

**Sensitivity analysis** re-solves every strategy across a range of available teams or demand (e.g.
0.5×–1.5×) and charts weighted unmet demand and coverage.

**Deep links**: `?page=overview|forecasts|planner|lab|strategies|methodology&scenario=normal|surge|shortage|disruption`.

---

## 13. The dashboard

* **Command overview** — scenario description, status pills (solver status, *constraints verified
  independently*, forecast model, synthetic data), KPI cards, demand trend with forecast band, plan
  summary, largest shortages, and a schematic map (marker size = demand, colour + shape + label = coverage
  status, arrows = moves) with an accessible table version.
* **Demand forecasts** — per-zone chart, forecast tables, model evaluation (validation and test tables,
  per-horizon and per-zone error, interval coverage), and a backtest view.
* **Allocation planner** — solver status and runtime, current vs recommended teams, gap table, movements,
  per-zone rationale, constraint checks for every strategy, and the map. A day-by-day view is shown in
  daily mode.
* **Scenario lab** — the override form, what changed, before/after metrics and per-zone differences.
* **Strategy evaluation** — the comparison table (hard constraints shown as Pass/FAIL), charts, fairness by
  zone and priority tier, sensitivity analysis and forecast error.
* **Methodology** — renders the docs.
* **Exports** (sidebar) — forecasts, plan, movements (and daily movements), strategy comparison (CSV), JSON
  run record (seed, model, horizon, solver status, parameters, metrics, timestamp), Markdown summary.

**Design**: dark charcoal theme, amber/red/green status colours never used alone (shapes and words too),
a persistent *decision-support prototype, synthetic data* banner, and loading, empty, infeasible and error
states. Verified visually in a browser; two chart bugs were found and fixed (see §17).

---

## 14. Evaluation results

Produced by `python scripts/run_evaluation.py` (seeds 42, 7, 2024, 99, 314). Raw files:
`results/evaluation_results.json`, `forecast_metrics.csv`, `strategy_comparison.csv`,
`summary_by_scenario.csv`. Synthetic data, so these describe the method on the generator, not real-world
performance.

### 14.1 Forecasting (held-out test, horizons 1–7, mean over 5 seeds)

| Model | MAE | RMSE | WAPE | Interval coverage (nominal 80%) |
|---|---|---|---|---|
| gradient_boosting | 4.653 | 6.630 | 19.6% | 75.9% |
| moving_average_28d | 4.687 | 6.709 | 19.7% | 78.4% |
| seasonal_average_3w | 5.011 | 7.092 | 21.1% | 81.4% |
| seasonal_naive | 6.226 | 8.618 | 26.1% | 79.4% |

Selected on validation: moving average in 4 of 5 seeds, gradient boosting in 1. The top two differ by
0.03 requests/day and the ranking flips between seeds, so **no claim is made that the complex model is
better**.

### 14.2 Single-period allocation (mean over 5 seeds)

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
| shortage + 30% floor | current | 429.4 | 611.4 | 76.2% | 0 | 0 | 0 | 52.0% |
| shortage + 30% floor | greedy | 366.6 | 491.5 | 80.5% | 81.4 | 3.6 | 0.6 | 52.9% |
| shortage + 30% floor | optimized | 366.6 | 493.7 | 80.5% | 64.0 | 3.0 | 0 | 54.4% |

* Optimizer status: **OPTIMAL in all 25 single-period runs**, each ≤ 0.13 s.
* **Greedy matched the optimizer’s weighted unmet demand in all 25 runs.** The optimizer’s measurable
  advantages: lower travel in 7 of 25 runs (e.g. shortage, seed 2024: 45 vs 77 team-minutes; surge, seed
  42: 169 vs 190), a proof of optimality, and correct handling of the equity floor (seed 42: greedy breaks
  the 30% floor in 3 zones; the optimizer satisfies it).
* **Fairness trade-off:** in the shortage scenario (seed 42), minimizing weighted unmet demand drops the
  worst zone from 35.6% (current) to 22.9%. A 30% floor raises it to 30.3% at a cost of +11.2 weighted
  unmet requests.

Single-scenario example (normal, seed 42): moving 3 teams (Northgate → Eastfield × 2, Old Town →
Southport × 1; 72 team-minutes) raises coverage from 89.1% to 96.2% and cuts weighted unmet demand from
361.9 to 123.0. Surge, seed 42: 6 teams move, weighted unmet 1,077.2 → 705.1, and 63 requests exceed zone
service caps (no plan can serve them).

### 14.3 Day-by-day allocation (mean over 5 seeds)

“Default” cases use the auto-selected forecast (usually the moving average, which is flat across the
week). “Seasonal” cases use the weekday-varying `seasonal_average_3w` forecast.

| Case | Optimizer wins / ties / worse | Weighted unmet: current → greedy → optimized | Travel: greedy vs optimized | Optimizer status |
|---|---|---|---|---|
| normal (default) | 0 / 5 / 0 | 120.6 → 26.6 → 26.6 | 29.2 vs 29.2 | 5 OPTIMAL |
| surge (default) | 2 / 3 / 0 | 457.9 → 193.2 → 192.4 | 114.8 vs 113.0 | 5 OPTIMAL |
| shortage (default) | 1 / 4 / 0 | 613.6 → 497.2 → 497.0 | 97.6 vs 86.2 | 5 OPTIMAL |
| disruption (default) | 0 / 5 / 0 | 120.6 → 31.1 → 31.1 | 31.4 vs 31.4 | 5 OPTIMAL |
| shortage + 30% floor | 0 / 4 / 1* | 613.6 → 497.2 → 499.1 | 97.6 vs 82.0 | 5 OPTIMAL |
| normal (seasonal) | 2 / 3 / 0 | 147.8 → 39.2 → 38.1 | 175.6 vs 207.6 | 4 OPTIMAL, 1 FEASIBLE† |
| surge (seasonal) | 2 / 3 / 0 | 521.8 → 256.1 → 255.1 | 302.8 vs 270.6 | 5 OPTIMAL |
| shortage (seasonal) | 5 / 0 / 0 | 656.9 → 541.3 → 537.4 | 317.0 vs 242.4 | 5 OPTIMAL |
| disruption (seasonal) | 2 / 3 / 0 | 147.8 → 46.4 → 45.5 | 208.8 vs 179.4 | 5 OPTIMAL |

\* The only “worse” run is where greedy breaks the equity floor (21 day-zone violations); the optimizer
satisfies it, which costs some weighted unmet demand.
† Stage 1 (weighted unmet) proved optimal; stage 2 (travel) was only FEASIBLE within the 10 s limit.

**Interpretation.** With the default flat forecast the optimizer wins 3 of 25 runs (0.77–2.85 weighted
requests, 0.06–2.0% of greedy’s value). With a weekday-varying forecast it wins 11 of 20 runs, including
all 5 shortage runs, by 0.30–6.96 weighted requests: mostly 0.4–1.5% of greedy’s value, and 17% in one run
where greedy left little unmet (normal, seed 7: 17.1 → 14.2). **Every gain is under 0.4% of total
demand.** Where demand varies by day, greedy keeps chasing each day and usually travels more; the
optimizer anticipates later days (in normal-seasonal it travels more because the objective buys extra
service first). The service-loss model is mild by construction (45-minute limit vs a 480-minute shift).

---

## 15. Testing, linting and CI

**110 pytest tests**, all passing locally and in CI (a clean install into a fresh virtual environment was
also verified, at the 69-test stage):

| File | Tests | Covers |
|---|---|---|
| `test_data.py` | 9 | determinism, schema, sizes, no negative demand, validation catches errors, CSV round-trip |
| `test_forecasting.py` | 11 | no future leakage (tampered future), chronological split, all zones/horizons, non-negative, metrics, constant/zero/short series |
| `test_optimizer.py` | 15 | brute-force optimality on small cases, conservation, integrality, caps, travel limit, blocked routes, infeasibility, zero resources/demand, lexicographic tie-break, λ behaviour, coverage floor, status reporting, validator |
| `test_baselines.py` | 6 | constraint compliance, determinism, greedy ≥ current, optimizer ≥ greedy, shared inputs |
| `test_simulation.py` | 12 | no mutation of the dataset, reset, surge/shortage/disruption propagate, multipliers, overrides, sensitivity |
| `test_metrics.py` | 7 | known metric values, zero/empty denominators, coverage bounds, fairness consistency, JSON export |
| `test_integration.py` | 4 | full workflow for every preset, reproducibility, infeasible scenario path, planning basis |
| `test_multiperiod.py` | 12 | brute force on 2-zone × 2-day cases, travel consuming service, conservation/staging/eligibility, validator, infeasible, zero cases, determinism, greedy validity and optimizer ≥ greedy, look-ahead beating myopic greedy, pipeline daily mode, per-day map matrix, daily sensitivity sweep |
| `test_app.py` | 11 | every page × scenario renders, disclaimer/status, Scenario Lab submit → INFEASIBLE → reset, sensitivity button, horizon control, deep links, daily mode on every page, daily overview/lab/sensitivity, what-if re-plan, robustness button, response-time standard |
| `test_response.py` | 9 | in-zone special case, neighbour coverage, capacity and weights, standard below local response, brute-force optimality, pipeline validity, unchanged default, daily rejection, robustness/what-if with a standard |
| `test_robustness.py` | 5 | per-sample scoring, correlated resampling, expected-outcome plan, validation/test residuals kept apart, full check |
| `test_counterfactual.py` | 6 | forced moves never beat the optimum, exact deltas, refusals with reasons, essential moves, every recommended move explained, bounds on ineligible routes |
| `test_scaled.py` | 3 | scaled-problem structure, single-period optimal at 50 zones, warm-started day-by-day never worse than greedy |

* **Ruff** lint and format: clean (line length 120).
* **Smoke test** (`scripts/smoke_test.py`): optimizer OPTIMAL, plan valid, all zones forecast, optimized ≤
  current.
* **GitHub Actions**: lint + tests + smoke test on Ubuntu, Python 3.12 and 3.13. Green on every pushed
  commit (latest code commit `2e0f92b`).

---

## 16. Responsible AI

* Decision support only; humans must review every plan. The banner says so on every page.
* All data synthetic; no real-world validation.
* Forecast uncertainty is substantial (~20% WAPE) and intervals are empirical.
* Priority weights (`1 + vulnerability proxy`) are a **value judgement**; they are editable and recorded in
  exports. The proxy is area-level and synthetic and must never be built from protected characteristics of
  individuals.
* **Priority weights do not guarantee fairness**: the shortage result shows the worst-off zone getting worse.
  The coverage floor makes equity an explicit, measurable constraint with a visible cost.
* Historical data can encode unequal service and under-reporting; geographic coverage is not equitable
  access.
* Full discussion: `docs/assumptions_and_limitations.md`.

---

## 17. Problems found and fixed along the way

| Problem | Fix |
|---|---|
| OR-Tools has no Windows-ARM64 wheels | Used the x64 Python 3.14 under emulation; documented |
| Service rate of 5 made “normal” a severe shortage | Raised to 10 requests/team/day; documented why |
| One surge case: CP-SAT FEASIBLE after 30 s with one worker | `linearization_level = 2` → OPTIMAL in 0.05 s, still deterministic |
| Map axes blew out to ±150 km on narrow screens | Fixed axis ranges with domain constraint |
| Forecast band drew stray markers | Band trace set to lines only |
| Bar labels clipped at chart top | `cliponaxis=False` |
| Unverified claim written into the evaluation doc (floor violations for the current allocation) | Checked against the CSV (it had none) and corrected |
| Headless Edge screenshots captured only a loading skeleton | Drove Edge with Playwright and waited for the app to finish |
| Day-by-day write-up misstated two percentages | Re-ran the evaluation, recomputed, corrected README and `docs/evaluation.md` |
| Local merge of PR #1 conflicted with the squash merge already on GitHub | Confirmed the trees were identical, replayed only the doc fix on top; nothing overwritten |
| Benchmark: day-by-day optimizer returned plans 700–2,500 weighted requests worse than greedy at 100 zones | Warm start from greedy, plus a guarantee: a valid warm-start plan is kept when the solver cannot beat it in time |
| Warm-start hints were added twice to the same variables (MODEL_INVALID in the travel stage) | Hints are cleared before the second stage; covered by a test |
| First robustness test scored plans on the same error windows used to build them | Plans are now built from validation errors only and scored on held-out test errors |
| A formula in `docs/mathematical_model.md` rendered as a line break | Fixed |

---

## 18. Repository, history and deliverables

* **Repository:** https://github.com/iwan-adler/resourcemind-ai (public, MIT licence).
* **Commit history on `main`:**
  1. `9d663f9` Build ResourceMind AI decision-support prototype
  2. `21ea398` Add README screenshots, CI badge and deep links
  3. `a00ec9b` Add optional day-by-day (multi-period) allocation mode (#1)
  4. `7b7e328` Correct day-by-day gain percentages in docs
  5. `5cc68d3` Add complete project report
  6. `2e0f92b` Make overview, scenario lab and sensitivity follow day-by-day mode (#2)
  7. `10c3e35` Update test counts and record PR #2 in the report
* **Screenshots:** `docs/images/` (overview, map, planner, forecasts, strategies), real captures of the app.
* **Pitch deck:** 11 slides with speaker notes, shared separately (exports to PowerPoint/PDF).
  Placeholders to fill:
  `[Team name]` on the cover and `[Streamlit URL]` on the closing slide.
* **Demo script:** `docs/demo_script.md`.
* **Judge Q&A preparation:** `docs/judge_qa.md` (18 likely questions with short answers, plus key numbers).

---

## 19. How to run everything

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py                    # http://localhost:8501
pytest                                  # 110 tests
python scripts/benchmark_scalability.py # results/scalability.csv
ruff check . && ruff format --check .
python scripts/smoke_test.py
python scripts/generate_data.py         # data/generated/*.csv
python scripts/run_evaluation.py        # results/*.json, *.csv
```

On Windows ARM64, create the virtual environment with an **x64** Python so OR-Tools installs.

---

## 20. Deployment status

**Not deployed yet.** Streamlit Community Cloud requires the repository owner to sign in and authorize it,
which could not be done on the owner’s behalf. Steps:

1. Sign in at https://share.streamlit.io with GitHub (`iwan-adler`).
2. **Create app → Deploy a public app from GitHub**: repo `iwan-adler/resourcemind-ai`, branch `main`,
   main file `app.py`.
3. **Advanced settings → Python 3.12.** No secrets are needed.
4. Deploy (the first build takes a few minutes), then add the live URL to the README and the deck.

---

## 21. Design decisions to defend to judges

* **Why optimization and not ML/RL for allocation?** The allocation has hard constraints (team counts,
  staging, routes) that must never be violated; an integer model guarantees them and can prove optimality.
  RL would need a simulator and gives no guarantees.
* **Why lexicographic?** Coverage of emergency demand comes first; travel only breaks ties. No arbitrary
  exchange rate between minutes and unserved people (a weighted mode exists if a planner wants one).
* **Why does greedy tie the optimizer?** With one decision per week the problem is close to a concave
  allocation problem, where marginal-gain greedy is often optimal, and the demo city is small. On
  25–100-zone benchmark cities the optimizer beats greedy in every run; with day-by-day planning, a
  response-time standard or policy constraints it also pulls ahead.
* **Why an independent validator?** So a solver bug or a modelling slip can never show a plan that breaks
  constraints; baselines get the same checks.
* **Why pick the forecast on validation, not test?** Choosing on test would leak and overstate accuracy.
* **Why empirical intervals?** Simple, reproducible and honest about their coverage; not oversold.
* **Why synthetic data?** No properly licensed public dataset with this structure was available; synthetic
  data is reproducible, safe to share and labelled everywhere.
* **How are weights chosen?** A documented default (`1 + vulnerability proxy`); editable and recorded.
  They are a value judgement and do not by themselves make a plan fair.

---

## 22. Limitations and future work

**Limitations:** synthetic data only; ~20% forecast error; teams treated as identical; a simplified
response-time rule (zone-centre travel, one local response time); no queues, backlog or crew-hour rules;
day-by-day planning does not prove optimality beyond about 10 zones within its time limit; the travel service-loss assumption (480-minute shift) is
simple; zone-level fairness metrics hide within-zone differences; Windows-ARM64 needs x64 Python.

**Future work (not built):**

1. Deploy to Streamlit Community Cloud and add the live link.
2. Richer response-time modelling (traffic, time of day, travel within zones).
3. A faster day-by-day solver for large cities (decomposition or parallel search).
4. Rolling re-planning (re-forecast and re-optimize each day as data arrives).
5. Validation on real, properly licensed data with domain experts, including an audit for under-reporting.
6. Stronger probabilistic forecasts (calibrated quantiles). A sample-based robust plan was tried and was
   not reliably better out of sample (see §11a).
