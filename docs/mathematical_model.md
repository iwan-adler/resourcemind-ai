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
* Single-period mode (default): demand within the horizon is aggregated; intra-week timing and dispatch
  queues are not modelled.
* Single-period mode: repositioning happens once, at the start of the horizon; travel time is a one-off
  cost. The day-by-day mode below relaxes both points.
* Coverage equal to 1 when demand is zero (nothing left unserved).

## Day-by-day (multi-period) mode

Optional (`planning_mode = "daily"` in the config, "Planning mode" in the app sidebar; code in
`resourcemind/allocation/multiperiod.py`). The single-period model above stays the default and is
unchanged. In daily mode teams may be repositioned every day, and travelling costs service time.

### Additional data

| Symbol | Meaning | Unit | Source in code |
|---|---|---|---|
| $d$ | day, $d = 1..H$ | – | – |
| $D_{id}$ | demand in zone $i$ on day $d$: daily forecast (`ZoneForecast.daily`, column `predicted`, or `upper` when planning against the upper interval) × stress multiplier, **rounded per day** | requests | `MultiPeriodProblem.demand` |
| $K_i^{day}$ | maximum useful service per day (`max_useful_service_per_day`) | requests/day | `service_cap` |
| $R^{day}$ | requests one team serves in a full day (default 10) | requests/team/day | `service_per_team` |
| $L$ | minutes in a team's daily shift, **assumed 480** (one 8-hour shift) | minutes | `minutes_per_shift` |

$A_i$, $M_i$, $w_i$, $T_{ji}$, $E_{ji}$, $r_i$ and $f$ are the same as in the single-period model.
Because demand is rounded per day, $\sum_d D_{id}$ can differ from the single-period $D_i$ by a few
requests.

### Decision variables (all integer)

* $X_{id}$ — teams stationed in zone $i$ at the end of day $d$; $X_{i0} = A_i$ is fixed.
* $y_{jid}$ — teams moved from $j$ to $i$ on day $d$ (only for $j 
e i$ with $E_{ji} = 1$).
* $S_{id} \in \{0..\min(D_{id}, K_i^{day})\}$, $U_{id} = D_{id} - S_{id}$.

### Constraints (for every zone $i$ and day $d$)

$$X_{id} = X_{i,d-1} + \sum_j y_{jid} - \sum_k y_{ikd} \qquad \text{(flow conservation)}$$
$$X_{i,d-1} - \sum_k y_{ikd} \ge \min(r_i, A_i) \qquad \text{(minimum retention: teams that stay)}$$
$$X_{id} \le M_i \qquad \text{(staging capacity, every day)}$$
$$S_{id} \le R^{day} X_{id} - \sum_j R^{day}\,\frac{T_{ji}}{L}\, y_{jid} \qquad \text{(service capacity after travel)}$$
$$S_{id} \ge \lceil f \cdot \min(D_{id}, K_i^{day}) \rceil \qquad \text{(only when } f > 0\text{; applies on every day)}$$

Teams leave only from where they were at the start of the day (outflow $\le X_{i,d-1}$, implied by the
retention row), so a team makes at most one hop per day.

**Travel consumes service.** A team that moves $j \to i$ on day $d$ spends $T_{ji}$ minutes of its
$L = 480$-minute shift driving and serves $R^{day}(1 - T_{ji}/L)$ requests in zone $i$ that day, then
full days afterwards. With $T^{max} = 45$ min and $R^{day} = 10$, one move costs at most
$10 \cdot 45/480 \approx 0.94$ requests. The capacity row is multiplied by $L$ so CP-SAT sees integers:
$L\,S_{id} \le R^{day} L X_{id} - \sum_j R^{day} T_{ji} y_{jid}$. Since $S_{id}$ is an integer, served
requests are the floor of the fractional capacity, which is exactly how the shared scoring function
(`compute_daily_outcome`) computes them for every strategy.

### Objective

Lexicographic as before: (1) minimise $\sum_d \sum_i w_i U_{id}$; (2) at that optimum, minimise
$\sum_d \sum_{j
e i} T_{ji}\, y_{jid}$ (team-minutes). The weighted-sum objective is also available.
Same solver settings (CP-SAT, one worker, fixed seed, `linearization_level = 2`) and the same status
semantics; infeasibility reasons reuse the single-period diagnosis (the static constraints of day 1 are
identical) plus a note when a per-day coverage floor is active.

### Baselines and validation

* **Keep current:** $y = 0$ on every day.
* **Day-by-day greedy:** day $d$ starts from the end-of-day-$(d-1)$ positions and applies the
  single-period greedy rule against that day's demand only (no look-ahead; it does not see the service
  lost while travelling, which the shared scoring then charges).
* **Validator** (`validate_daily_plan`): recomputes, per day and from raw data, flow conservation of the
  reported stationed counts against the reported moves, total teams, minimum retention, staging capacity,
  route eligibility, integrality, non-negativity and the per-day coverage floor.
* All three are scored by `compute_daily_outcome`; the comparison table has the same columns as in
  single-period mode, with coverage and fairness computed from per-zone horizon totals.

### Limits of this extension

* Teams arrive within the day they move; there are no multi-day trips, crew hours or rest rules.
* Demand on a day is served only that day (no backlog carried over).
* The plan reacts to the forecast, not to realised demand; it is not re-optimised on a rolling basis.
