# Judge Q&A preparation

Short, defensible answers to the questions judges are most likely to ask. Every number is from
`docs/evaluation.md` / `results/` (synthetic data, mean over 5 seeds unless stated). Rule of thumb: answer
in one or two sentences, give the number, then admit the limit.

---

## The data

**1. Is any of this real data?**
No. Everything is synthetic, generated locally from a fixed seed, and it’s labelled that way on every page.
No properly licensed public dataset had zones, daily demand, team positions and travel times together, and
synthetic data is reproducible and safe to share.

**2. Then how do you know it works in reality?**
We don’t claim it does. What we can show is that the *method* is sound: no leakage, constraints verified
independently, honest baselines. Real use would need validation on real, licensed data with domain experts.

**3. Is the synthetic data just designed to make you look good?**
The opposite happened: on it, our simple greedy baseline tied the optimizer in all 25 single-period runs,
and we report that. The one deliberate choice is the legacy staffing rule (teams in proportion to
population, not demand), which creates the mismatch the planner is meant to fix.

## Forecasting

**4. How do you know there’s no data leakage?**
Features only use data up to the forecast day, and the split is chronological: train → 28 validation days
→ final 28 test days. A test feeds deliberately corrupted *future* values and checks the forecast does not
change. The default model is chosen on validation; test results are only reported.

**5. Why is your machine-learning model no better than a moving average?**
On this data they’re tied: test MAE 4.65 vs 4.69 requests/day, and the ranking flips between seeds. Weekly
patterns are mild relative to random noise, which favours smoothing. Both beat the seasonal naive baseline by
about 25%. We chose not to oversell the complex model.

**6. How good are the uncertainty intervals?**
They’re empirical (from validation errors), not guaranteed probabilities. Measured on test, the nominal 80%
band covered 76–81% of actual values, so it slightly under-covers.

## Optimization

**7. Why an optimizer instead of machine learning or reinforcement learning?**
Allocation has hard rules (team counts, staging space, travel limits) that must never be broken. An integer
model guarantees them and can prove the plan is optimal; RL would need a simulator and gives no guarantee.
Forecasting is where learning helps, so that’s where we use it.

**8. Your greedy baseline matched the optimizer. Why bother?**
On our 10-zone demo city, yes, it tied in all 25 runs, and we report it. That’s a small-problem effect:
on synthetic cities of 25, 50 and 100 zones the optimizer beat greedy in every run, by up to 141 weighted
requests at 100 zones. Even at 10 zones it (a) used less travel in 7 of 25 runs, (b) *proves* optimality,
and (c) respects policy rules greedy can’t: with a 30% equity floor, greedy broke it in 3 zones. With a
20-minute response-time standard it beat greedy in 5 of 20 runs and moved far fewer teams.

**9. What exactly does the optimizer minimize?**
First, weighted unmet demand; then, without making that any worse, total team-minutes of travel. That
order means travel can never buy a worse service outcome, and we never had to invent an exchange rate
between minutes and unserved people. A weighted version with an explicit trade-off parameter is available.

**10. What happens if no valid plan exists?**
The solver says INFEASIBLE and no plan is shown. The app explains why (e.g. “32 teams must be stationed but
staging capacity is 20”) and computes the smallest capacity increase that would fix it. A plan that is only
FEASIBLE (not proven optimal) is labelled as such.

**11. How do you know a plan doesn’t break the rules?**
Every plan, from the optimizer and the baselines, is re-checked by an independent validator that recomputes
every constraint from the raw data without trusting the solver. All 3 strategies are scored by the same
function on the same inputs.

**12. How fast is it, and does it scale?**
We benchmarked synthetic cities up to 100 zones (226 teams). Single-period planning stays proven optimal in
about 1 s. Day-by-day planning is the limit: from 25 zones it can’t prove optimality within 20 s per stage
(at most 3% from the proven bound at 25 zones, 6% at 50). At 100 zones it doesn’t improve on greedy in time, so it
keeps the greedy plan. We found that the hard way: before we added that guarantee, it returned plans worse
than greedy at 100 zones.

## Fairness and ethics

**13. How did you choose the priority weights? Isn’t that biased?**
Default: 1 + a synthetic, area-level vulnerability index (range 1–2). It’s a value judgement, so it’s
documented, editable in the app and recorded in every export. It must never be built from protected
characteristics of individuals; in a real setting it should be set with the communities affected.

**14. Does the plan help the most vulnerable areas?**
Not automatically, and we show it. In the severe shortage scenario, minimizing weighted unmet demand drops the
worst-covered zone from 35.6% to 22.9%. A coverage floor (a hard equity rule) raises it to 30.3% at a cost of
11.2 extra weighted unmet requests. The tool makes that trade-off explicit so a human decides.

**15. Could this be used in a real emergency tomorrow?**
No. It’s a decision-support prototype: synthetic data, ~20% daily forecast error, a simplified response-time
rule and no queue modelling. A person must review every plan. The banner on every page says so.

## Scope and next steps

**16. What does “coverage” mean here?**
By default, service capacity matched to forecast request volume in each zone. With the optional
response-time standard, demand only counts if a team can reach it within, say, 20 minutes, and a team in a
neighbouring zone can help. It uses zone-centre travel times and one synthetic local response time
(11.1 min), so it’s a coverage rule, not a dispatch simulation.

**16b. What if the forecast is wrong?**
We tested it: each plan is scored against 200 demand outcomes built from the model’s real errors on the
held-out test period. The optimized plan keeps its advantage (normal: 7% vs 21% chance of a sub-90% week
compared with today’s allocation). Interestingly, planning against a pessimistic forecast, or optimizing for
the average outcome, was *not* reliably better than planning for the point forecast.

**16c. Why did the plan move that team? Why not another one?**
The app answers that directly: forbid a recommended move or force a different one, and it re-solves the real
model and shows the exact cost. For example, forcing a team from Harbor District to Old Town would cost 21.5
weighted requests; forbidding either recommended move leaves an equally good plan.

**17. What would you do with more time?**
Rolling re-planning each day as new data arrives; a faster day-by-day solver for large cities; richer
response-time modelling (traffic, time of day); validation on real, licensed data with an audit for
under-reporting; calibrated probabilistic forecasts.

**18. What did it cost to build and run?**
Nothing: open-source Python (scikit-learn, OR-Tools, Streamlit), no paid APIs or keys, no external AI
service, and it runs fully offline. One command reproduces every number in the deck.

---

## Numbers to know by heart

| Fact | Number |
|---|---|
| Coverage, normal week (seed 42): current → plan | 89.1% → 96.2% (3 teams moved) |
| Weighted unmet, normal (mean of 5 seeds): current → plan | 118.5 → 24.6 |
| Forecast test MAE: gradient boosting / moving average / naive | 4.65 / 4.69 / 6.23 |
| Greedy ties optimizer (single-period) | 25 of 25 runs |
| Optimizer wins (day-by-day, weekday-varying forecast) | 11 of 20 runs, all gains < 0.4% of demand |
| Worst zone in shortage: current / optimized / with 30% floor | 35.6% / 22.9% / 30.3% |
| Optimizer status (single-period) | OPTIMAL in 25 of 25 |
| Optimizer vs greedy at 25–100 zones (single-period) | ahead in every run, up to 141 weighted requests |
| Response standard 20 min: optimizer wins | 5 of 20 runs, never worse |
| Automated tests | 110, CI on Python 3.12 and 3.13 |
