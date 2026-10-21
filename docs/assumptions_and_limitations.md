# Assumptions, limitations and responsible use

## What this is

ResourceMind AI is a **decision-support prototype**. It recommends how to reposition response teams for a
planning horizon and explains why. It is **not** an autonomous dispatch system, has **not** been validated
against real emergency operations, and its outputs must not be treated as operational instructions.
Any real use would need human review, domain validation, and governance.

## Data

* All data are **synthetic** (see `data_dictionary.md`). Patterns such as weekly peaks and spikes are
  assumptions built into the generator, so results describe the method on that generator only.
* Real incident data would contain reporting delays, missing records, changing definitions and
  **historically unequal service patterns**. A forecaster trained on such data learns *recorded* demand,
  which can under-state need where people under-report or were under-served. That could steer resources
  away from communities that already receive less.

## Forecasting

* Forecast error is substantial (WAPE ≈ 20% daily on the synthetic test set).
* Intervals are **empirical** quantiles of validation residuals, not guaranteed probabilistic intervals;
  their measured test coverage was 76–81% against a nominal 80%.
* The "upper interval" planning basis is a simple cautious option, not a calibrated risk model.
* Scenario stress multipliers are explicit what-if assumptions and are labelled as such.

## Optimization

* Teams are identical and handle a fixed number of requests per day everywhere.
* Demand is aggregated over the horizon; timing within the week, response-time targets and queueing
  are not modelled. Coverage here means *service capacity matched to forecast volume*, not response time.
* Repositioning is a single move at the start of the horizon in the default single-period mode; travel time
  is a cost, not lost service time. The optional day-by-day mode moves teams daily and deducts travel time
  from a 480-minute shift (an assumption), but has no backlog, crew-hour rules or rolling re-planning.
* Maximum useful service (K) and staging capacity (M) are synthetic parameters.
* The optional response-time standard uses zone-centroid travel times and one synthetic local response
  time for every zone (about 11.1 min on the demo map, from map geometry). Real response times vary
  within a zone, by time of day and by traffic; this is a coverage rule, not a dispatch simulation. The
  greedy baseline does not reason about cross-zone response, which favours the optimizer in that mode.
* The robustness check resamples about 20 complete forecast windows from the held-out test period, so it
  shows how plans behave under the errors this model actually made, not a full probability model.
* Scalability was measured on synthetic problems up to 100 zones. Single-period planning stayed fast and
  proven optimal; day-by-day planning stops proving optimality from 25 zones within a 20 s limit per stage
  (see `docs/evaluation.md`).

## How priority weights are chosen

Default policy: $w_i = 1 + \text{vulnerability\_index}_i$ (range 1–2). The vulnerability index is a
synthetic, area-level proxy; it does not use, and must not be built from, protected characteristics of
individuals. Weights are a **value judgement**: whoever sets them decides whose unmet demand counts more.
The Scenario Lab lets users change them and shows the effect, and changed weights are recorded in the run
export.

## Fairness

* **Priority weights do not guarantee fairness.** Minimising total weighted unmet demand can move teams
  away from a low-demand zone and leave it badly covered. In the shortage scenario (seed 42), the
  worst-covered zone falls from 35.6% (current) to 22.9% (optimized) while total weighted unmet demand
  improves.
* The optional **coverage floor** turns equity into an explicit hard constraint. With a 30% floor, worst-zone
  coverage rose to 30.3% at a cost of +11.2 weighted unmet requests. If the floor cannot be met, the solver
  reports INFEASIBLE rather than quietly ignoring it.
* **Geographic coverage ≠ equitable access.** Zone-level averages hide differences within zones (mobility,
  language, digital access, trust). Zone-level metrics are a starting point for review, not proof of equity.
* Reported fairness measurements: coverage per zone, worst-zone coverage, coverage range, Gini of unmet
  demand, and coverage by priority tier.

## Responsible-use checklist for any real pilot

1. Validate forecasts and capacities against local operational data and expert judgement.
2. Audit historical data for under-reporting and unequal service before training.
3. Set and publish the weighting policy with community and stakeholder input.
4. Keep a human decision-maker accountable for every plan; log overrides.
5. Monitor zone- and group-level outcomes after deployment, not just totals.
