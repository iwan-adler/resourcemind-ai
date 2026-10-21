# Demo script (≈ 6 minutes)

Start: `streamlit run app.py`, then open http://localhost:8501. The first load takes a few seconds (forecast
evaluation); later interactions are cached.

1. **Command overview — normal operations (45 s).**
   Point out the synthetic-data banner and status pills (`Solver: OPTIMAL`, *Constraints
   validated*). Read the cards: about 2,226 requests forecast for 7 days, 32 teams. Moving 3 teams raises
   coverage from 89.1% to 96.2% and cuts weighted unmet demand from 361.9 to 123.0. Scroll to the map and
   toggle *Current allocation* ↔ *Recommended plan*; open the accessible table.
2. **Demand forecasts (45 s).** Select Eastfield (Z05). Show the forecast band, then the *Model evaluation*
   tab: chronological split, validation used for selection, test reported only. Note that gradient boosting
   and the moving average are effectively tied, so we do not claim the complex model wins.
3. **Allocation planner (60 s).** Show current vs recommended bars against staging capacity, the
   *Movements* tab (Northgate → Eastfield ×2, Old Town → Southport ×1) and *Why this plan*. The
   *Constraint checks* tab shows all strategies pass the independent validator.
   Open **What if?**: forbidding either recommended move costs nothing (an equally good plan exists). Then
   choose Harbor District → Old Town and press **Re-plan with this move**: the re-solved plan is 21.5
   weighted requests worse. Every answer is a real re-solve, not a guess.
   Finally open *Planning settings* and tick **Apply a response-time standard** (20 min): the plan changes
   to one move (Eastfield → Westbrook) and weighted unmet falls from 52.4 to 20.0, because a team in a
   neighbouring zone can cover part of the demand. Untick it again before moving on.
4. **Switch scenario to "Sudden demand surge" (30 s).** The history changes (last 7 days × 1.9), so the
   forecast rises and the plan changes (6 teams move). The system-wide shortage message appears.
5. **Scenario lab — honest infeasibility (45 s).** Set *Staging capacity* to 0.3 and press **Run
   simulation**. The optimizer reports **INFEASIBLE**, explains why (32 teams but less staging space) and
   gives the smallest capacity increase that would fix it. The current allocation is shown with its
   violations. Press **Reset to preset defaults**.
6. **Fairness trade-off (45 s).** Choose *Severe resource shortage*. In *Strategy evaluation*, the zone
   coverage chart shows the optimizer lowering the worst zone. In the Scenario lab, set *Coverage floor* to
   0.30 and run: the worst zone rises to 30%, weighted unmet rises slightly, and greedy now shows floor
   violations.
7. **Robustness, sensitivity and export (45 s).** Switch back to *Normal operations*. In *Strategy
   evaluation*, press **Run robustness check**: every plan scored against 200 outcomes built from the model's real
   forecast errors (normal: 7% vs 21% chance of a sub-90% week compared with today's allocation). Then *Run
   sensitivity analysis* (teams 0.5×–1.5×) and download the JSON run record from *Export results*.
8. **If asked about scale:** the optimizer ties greedy on this 10-zone city but beats it in every run on
   25–100-zone benchmark cities (`docs/evaluation.md`, *Scalability*).

Key message: real forecasting, a real constrained optimizer, independent validation, explanations you can
interrogate, and honest reporting, including where the simple greedy baseline is just as good.
