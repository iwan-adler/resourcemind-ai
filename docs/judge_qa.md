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
With one move per week, the problem is close to a type where greedy is naturally optimal, so the tie is
expected, and we report it. The optimizer still (a) used less travel in 7 of 25 runs, (b) *proves*
optimality, and (c) respects policy rules greedy can’t: with a 30% equity floor, greedy broke it in 3 zones.
In day-by-day planning the optimizer beats greedy in 11 of 20 runs with a weekday-varying forecast, though
every gain is under 0.4% of total demand.

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
Single-period: ≤ 0.13 s per solve (10 zones). Day-by-day: about 0.3–3.4 s on average, worst case 14.7 s.
We have only tested 8–12 zones; larger cities would need testing, and possibly time limits or decomposition.

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
No. It’s a decision-support prototype: synthetic data, ~20% daily forecast error, no response-time or queue
modelling. A person must review every plan. The banner on every page says so.

## Scope and next steps

**16. What does “coverage” mean here?**
Service capacity matched to forecast request volume over the horizon, not response time. Response-time
targets are our main next step.

**17. What would you do with more time?**
Response-time targets; rolling re-planning each day as new data arrives; validation on real, licensed data
with an audit for under-reporting; calibrated probabilistic forecasts.

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
| Automated tests | 84, CI on Python 3.12 and 3.13 |
