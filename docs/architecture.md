# Architecture

```
configs/default_scenario.json ──► config.AppConfig
                                        │
data/generate.py ──► Dataset (zones, history, travel)   [read-only]
data/validate.py ──► schema / range / key checks
                                        │
simulation/scenarios.py ──► ScenarioInputs  (copies + preset & user overrides, re-validated)
                                        │
forecasting/features.py ──► leakage-safe (zone, origin, horizon) rows
forecasting/models.py   ──► seasonal naive, seasonal avg, moving avg, gradient boosting
forecasting/evaluation.py ─► chronological eval + empirical intervals + future forecast
                                        │
simulation/scenarios.py ──► planning demand (forecast × stress) ──► AllocationProblem
                                        │
allocation/baselines.py  ──► current, greedy           ┐
allocation/optimizer.py  ──► CP-SAT (lexicographic)    ├─► allocation/models.compute_outcome (shared scoring)
                                                       ┘   allocation/validation.validate_allocation (independent)
allocation/multiperiod.py ─► optional day-by-day CP-SAT ┐
allocation/baselines.py  ──► current_daily, greedy_daily ├─► multiperiod.compute_daily_outcome (shared scoring)
                                                       ┘   allocation/validation.validate_daily_plan (independent)
allocation/explain.py    ──► per-zone rationale, plan summary
metrics/evaluation.py    ──► forecast metrics, coverage, fairness
simulation/sensitivity.py ─► re-solve all strategies across a parameter range
                                        │
pipeline.py ──► run_pipeline(): the single workflow used by the app, the evaluation script and tests
                                        │
app.py (Streamlit) + visualization/charts.py (Plotly) + ui/components.py (styling)
scripts/: generate_data.py · run_evaluation.py · smoke_test.py
```

## Design decisions

* **One pipeline function.** `run_pipeline` is the only path from inputs to results, so the UI cannot show
  numbers the tests and evaluation do not also produce.
* **Separation of layers.** History, forecast and stress-tested demand are separate objects. Scenario
  overrides are a frozen dataclass applied to *copies* of the dataset; resetting means discarding them.
* **Shared scoring and validation.** Every strategy is scored by `compute_outcome` and checked by
  `validate_allocation`; the optimizer's own objective is never trusted for reporting.
* **Determinism.** Fixed seeds everywhere, one CP-SAT worker, no early-stopping split in gradient boosting.
* **Caching (UI).** `st.cache_data` keyed on the JSON-serialised config, scenario, model and overrides; any
  parameter change produces a new key, so stale results cannot be shown.
* **Offline by design.** No map tiles, APIs, LLMs or downloads at runtime. The map is a schematic Plotly scatter.
* **No statsmodels.** A statistical time-series model was optional; evaluating it with daily rolling origins
  would need hundreds of refits per run. It was left out to keep the dependency set and startup time small.
