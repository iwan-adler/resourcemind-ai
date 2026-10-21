# Mathematical model

ResourceMind AI positions **response teams** across **zones** for a planning horizon of
`H` days (default 7). Demand is measured in **service requests**; resources are measured in
**teams**. The two units are linked by an explicit service rate.

## Sets, data and units

| Symbol | Meaning | Unit | Source in code |
|---|---|---|---|
| $i, j$ | zones ($i$ = receiving zone, $j$ = origin zone) | – | `AllocationProblem.zone_ids` |
| $D_i$ | planning demand over the horizon (forecast × stress multiplier, rounded) | requests | `demand` |
| $A_j$ | teams currently stationed at origin $j$ (after scenario changes) | teams | `available` |
| $M_i$ | staging capacity: max teams that can be based in zone $i$ | teams | `staging_capacity` |
| $K_i$ | maximum useful service in zone $i$ over the horizon (e.g. hospital intake/access limit) | requests | `service_cap` |
| $R$ | requests one team can handle over the horizon = rate/day × $H$ (default 10 × 7 = 70) | requests/team | `service_per_team` |
| $w_i$ | priority weight (default $1 + $ vulnerability proxy, range 1–2) | dimensionless | `weights` |
| $T_{ji}$ | travel time to reposition one team from $j$ to $i$ ($T_{jj}=0$) | minutes | `travel` |
| $E_{ji}$ | route eligible: $T_{ji} \le T^{max}$, not blocked, neither zone inaccessible; $E_{jj}=1$ | boolean | `eligible` |
| $r_j$ | minimum teams that must remain at origin $j$ (default 1, capped at $A_j$) | teams | `retain` |
| $f$ | optional coverage floor (equity policy), 0 = off | share | `coverage_floor` |

## Decision variables (all integer)

* $x_{ji} \in \{0,\dots,A_j\}$ — teams from origin $j$ stationed in zone $i$ (created only where $E_{ji}=1$).
  The code stores this as `x[j, i]`; $x_{jj}$ is "teams that stay".
* $X_i = \sum_j x_{ji}$ — teams stationed in zone $i$.
* $S_i \in \{0,\dots,\min(D_i, K_i)\}$ — requests served in zone $i$.
* $U_i = D_i - S_i$ — unmet requests in zone $i$.

## Constraints

$$\sum_i x_{ji} = A_j \quad \forall j \qquad \text{(conservation: every team is somewhere)}$$
$$x_{jj} \ge \min(r_j, A_j) \quad \forall j \qquad \text{(minimum local presence)}$$
$$X_i \le M_i \quad \forall i \qquad \text{(staging capacity)}$$
$$S_i \le R \cdot X_i,\quad S_i \le D_i,\quad S_i \le K_i \quad \forall i \qquad \text{(service capacity)}$$
$$U_i = D_i - S_i \ge 0$$
$$x_{ji} = 0 \text{ if } E_{ji} = 0 \qquad \text{(travel limit, blocked routes, inaccessible zones)}$$
$$S_i \ge \lceil f \cdot \min(D_i, K_i) \rceil \quad \forall i \qquad \text{(only when the coverage floor } f > 0)$$

Conservation is an equality (not $\le$): teams cannot disappear, so "keep everyone where they are" is
always a candidate unless staging capacities or the coverage floor make it illegal.

## Objective

**Lexicographic (default).**

1. Minimise weighted unmet demand $\;Z_1 = \sum_i w_i U_i$.
2. Add the constraint $\sum_i w_i U_i \le Z_1^*$ and minimise travel $\;Z_2 = \sum_{j \ne i} T_{ji}\,x_{ji}$ (team-minutes).

This avoids choosing an arbitrary exchange rate: travel can never buy a worse service outcome, and
among equally good plans the planner picks the one with the least repositioning.

**Weighted sum (optional, exposed in the UI).**

$$\min \sum_i w_i U_i + \lambda \sum_{j\ne i} T_{ji} x_{ji}$$

$\lambda$ has units of *weighted requests per team-minute*. The default $\lambda = 0.01$ means 100
team-minutes of travel are worth one weighted unserved request. One team serves $R = 70$ requests over
the horizon, so a 30-minute move "costs" 0.3 weighted requests: movement only blocks marginal moves, it
cannot overwhelm coverage. Setting $\lambda$ very high (e.g. 5) makes most moves not worthwhile —
the tests check this behaviour.

**Integer scaling.** CP-SAT needs integer coefficients, so $w_i$ and $\lambda$ are multiplied by 1000
(`WEIGHT_SCALE`). Weights are therefore resolved to 0.001. All reported values are recomputed from the
returned allocation in real units.

## Solver and status semantics

Google OR-Tools **CP-SAT**, one worker, fixed random seed, `linearization_level = 2` (a tighter LP
relaxation; it turned a case that could not prove optimality in 30 s into one proven optimal in 0.05 s).

| Status | Meaning in ResourceMind AI |
|---|---|
| `OPTIMAL` | Both stages proven optimal. |
| `FEASIBLE` | A valid plan was found but optimality was not proven within the time limit (either stage). Shown with a warning. |
| `INFEASIBLE` | No allocation satisfies all hard constraints. No plan is shown; the app lists reasons and the smallest staging-capacity increase that would restore feasibility (an elastic re-solve). |
| `UNKNOWN` | Time limit reached without any solution. No plan is shown. |
| `MODEL_INVALID` | Model construction error (should not occur; reported if it does). |
| `INVALID` | Assigned by the app if a solver plan ever fails independent validation. The plan is rejected. |

Every returned allocation (optimizer *and* baselines) is re-checked by
`resourcemind.allocation.validation.validate_allocation`, which recomputes conservation, retention,
route eligibility, staging capacity, integrality, non-negativity and the coverage floor from the raw
data without trusting the solver.

## Baselines (same inputs, same scoring)

* **Current allocation:** $x_{jj} = A_j$.
* **Greedy:** repeatedly move one not-yet-moved team along the eligible route with the highest
  $\dfrac{\text{gain}_i - \text{loss}_j}{1 + T_{ji}/60}$, where gain/loss are the change in weighted
  served requests. Stops when no move has positive net gain. Respects staging capacity, retention and
  route eligibility; it does **not** know about the coverage floor, and validation reports that honestly.

## Explanations

* **Limiting factor** per zone (binding constraint under the plan): demand fully served, service cap $K$,
  staging capacity $M$, or team capacity $R\cdot X$.
* **Rationale** per zone: why it gained, released or kept teams, derived from the computed numbers.

## Gap analysis (before allocation)

$\text{gap}_i = D_i - R \cdot A_i$ (requests). Positive = shortage, negative = spare capacity.
Coverage before allocation $= \min(D_i, K_i, R A_i) / D_i$.

## Key assumptions

* Teams are interchangeable and serve $R$ requests per day regardless of zone.
* Demand within the horizon is aggregated; intra-week timing and dispatch queues are not modelled.
* Repositioning happens once, at the start of the horizon; travel time is a one-off cost.
* Coverage equal to 1 when demand is zero (nothing left unserved).
