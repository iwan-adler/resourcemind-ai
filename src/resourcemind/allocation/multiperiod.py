"""Day-by-day (multi-period) team repositioning with Google OR-Tools CP-SAT.

The single-period model (optimizer.py) makes ONE repositioning decision for the
whole horizon. This model plans every day d = 1..H: teams may move each day,
and a team that moves loses the part of that day's shift spent travelling.

Decision variables (all integer):
    X[i, d]     teams stationed in zone i at the end of day d (X[i, 0] = A_i, fixed)
    y[j, i, d]  teams moved from zone j to zone i on day d (only eligible routes, j != i)
    S[i, d]     service requests delivered in zone i on day d
    U[i, d]     = D[i, d] - S[i, d], unmet requests

Constraints, for every zone i and day d:
    X[i, d] = X[i, d-1] + sum_j y[j, i, d] - sum_k y[i, k, d]      flow conservation
    X[i, d-1] - sum_k y[i, k, d] >= min(retain_i, A_i)              minimum retention
    X[i, d] <= M_i                                                  staging capacity
    S[i, d] <= min(D[i, d], K_i)                                    (variable domain)
    S[i, d] <= R * X[i, d] - sum_j R * T_ji / SHIFT * y[j, i, d]    service capacity

Travel consumes service: a team that moves j -> i on day d spends T_ji minutes of
its SHIFT-minute shift driving, so it serves R * (1 - T_ji / SHIFT) requests in
zone i that day (assumption: one 480-minute shift per team per day). The service
row is multiplied by SHIFT so CP-SAT sees integer coefficients; ``S`` is an
integer, so served = floor of the fractional capacity, exactly as in
``compute_daily_outcome``. Teams only leave from where they were at the start
of the day, so a team moves at most one hop per day.

Objectives (lexicographic, as in the single-period model):
    (1) minimise sum_d sum_i w_i U[i, d]
    (2) minimise sum_d sum_{j != i} T_ji y[j, i, d] (team-minutes) at the stage-1 optimum
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from resourcemind.allocation.models import AllocationProblem
from resourcemind.allocation.optimizer import _STATUS, WEIGHT_SCALE, _solver, diagnose_infeasibility

MINUTES_PER_SHIFT = 480  # one 8-hour shift per team per day


@dataclass(frozen=True)
class MultiPeriodProblem:
    zone_ids: tuple[str, ...]
    zone_names: tuple[str, ...]
    priority_tiers: tuple[str, ...]
    demand: np.ndarray  # D[i, d], integer service requests per zone and day, shape (n, H)
    available: np.ndarray  # A_i, teams in zone i before day 1
    staging_capacity: np.ndarray  # M_i, max teams stationed in zone i on any day
    service_cap: np.ndarray  # K_i, max useful service requests in zone i per DAY
    weights: np.ndarray  # w_i
    travel: np.ndarray  # T[j, i], minutes
    eligible: np.ndarray  # bool[j, i]
    retain: np.ndarray  # minimum teams that must stay in zone i every day
    service_per_team: int  # R, service requests per team per DAY
    max_travel_minutes: int
    minutes_per_shift: int = MINUTES_PER_SHIFT
    # Equity policy applied per day: S[i, d] >= ceil(floor * min(D[i, d], K_i)).
    coverage_floor: float = 0.0

    @property
    def n(self) -> int:
        return len(self.zone_ids)

    @property
    def horizon(self) -> int:
        return int(self.demand.shape[1])

    @property
    def servable(self) -> np.ndarray:
        """min(D[i, d], K_i), shape (n, H)."""
        return np.minimum(self.demand, self.service_cap[:, None])

    @property
    def floor_requirement(self) -> np.ndarray:
        return np.ceil(self.coverage_floor * self.servable - 1e-9).astype(int)

    @property
    def keep(self) -> np.ndarray:
        return np.minimum(self.retain, self.available).astype(int)

    def day_problem(self, d: int, available: np.ndarray) -> AllocationProblem:
        """Single-period problem for day d (0-based) starting from ``available`` teams."""
        return AllocationProblem(
            zone_ids=self.zone_ids,
            zone_names=self.zone_names,
            priority_tiers=self.priority_tiers,
            demand=self.demand[:, d].astype(int),
            available=np.asarray(available, dtype=int),
            staging_capacity=self.staging_capacity,
            service_cap=self.service_cap,
            weights=self.weights,
            travel=self.travel,
            eligible=self.eligible,
            retain=self.retain,
            service_per_team=int(self.service_per_team),
            horizon_days=1,
            max_travel_minutes=self.max_travel_minutes,
            coverage_floor=self.coverage_floor,
        )


def daily_problem_from(problem: AllocationProblem, demand_daily, service_cap_daily, rate_daily: int):
    """Multi-period problem sharing zones, teams, network and policy with ``problem``."""
    return MultiPeriodProblem(
        zone_ids=problem.zone_ids,
        zone_names=problem.zone_names,
        priority_tiers=problem.priority_tiers,
        demand=np.maximum(np.asarray(demand_daily, dtype=int), 0),
        available=problem.available,
        staging_capacity=problem.staging_capacity,
        service_cap=np.asarray(service_cap_daily, dtype=int),
        weights=problem.weights,
        travel=problem.travel,
        eligible=problem.eligible,
        retain=problem.retain,
        service_per_team=int(rate_daily),
        max_travel_minutes=problem.max_travel_minutes,
        coverage_floor=problem.coverage_floor,
    )


@dataclass
class DailyPlan:
    stationed: np.ndarray  # X[i, d] for d = 1..H, shape (n, H)
    moves: np.ndarray  # y[d, j, i], shape (H, n, n), zero diagonal

    @classmethod
    def from_moves(cls, available: np.ndarray, moves: np.ndarray) -> DailyPlan:
        """Derive stationed teams from moves by flow conservation."""
        moves = np.asarray(moves, dtype=int)
        x = np.asarray(available, dtype=int).copy()
        cols = []
        for d in range(moves.shape[0]):
            x = x + moves[d].sum(axis=0) - moves[d].sum(axis=1)
            cols.append(x.copy())
        stationed = np.stack(cols, axis=1) if cols else np.zeros((len(x), 0), dtype=int)
        return cls(stationed=stationed, moves=moves)


@dataclass
class DailyResult:
    strategy: str
    status: str  # OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN / MODEL_INVALID / HEURISTIC / BASELINE
    plan: DailyPlan | None
    runtime_seconds: float
    messages: list[str] = field(default_factory=list)
    stage_statuses: dict[str, str] = field(default_factory=dict)
    objective: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_solution(self) -> bool:
        return self.plan is not None


@dataclass
class DailyOutcome:
    stationed: np.ndarray  # (n, H)
    capacity: np.ndarray  # (n, H) floor of team service capacity after travel loss
    served: np.ndarray  # (n, H)
    unmet: np.ndarray  # (n, H)
    weighted_unmet: float
    travel_minutes: float
    teams_moved: int  # team-moves summed over days

    @property
    def served_total(self) -> np.ndarray:
        return self.served.sum(axis=1)

    @property
    def unmet_total(self) -> np.ndarray:
        return self.unmet.sum(axis=1)


def daily_capacity(problem: MultiPeriodProblem, plan: DailyPlan) -> np.ndarray:
    """floor(R * X[i, d] - sum_j R * T_ji / SHIFT * y[j, i, d]), never below 0."""
    R, shift = int(problem.service_per_team), int(problem.minutes_per_shift)
    lost = np.einsum("dji,ji->id", plan.moves, problem.travel.astype(int)) * R  # (n, H), x SHIFT
    scaled = R * shift * plan.stationed - lost
    return np.maximum(scaled // shift, 0)


def compute_daily_outcome(problem: MultiPeriodProblem, plan: DailyPlan) -> DailyOutcome:
    """Shared scoring for every daily strategy: S[i, d] = min(D, K, capacity after travel)."""
    capacity = daily_capacity(problem, plan)
    served = np.minimum(problem.servable, capacity)
    unmet = problem.demand - served
    return DailyOutcome(
        stationed=plan.stationed,
        capacity=capacity,
        served=served,
        unmet=unmet,
        weighted_unmet=float((problem.weights[:, None] * unmet).sum()),
        travel_minutes=float(np.einsum("dji,ji->", plan.moves, problem.travel.astype(int))),
        teams_moved=int(plan.moves.sum()),
    )


def daily_movements_table(problem: MultiPeriodProblem, plan: DailyPlan) -> pd.DataFrame:
    """One row per day and origin -> destination move with teams > 0."""
    rows = []
    shift = problem.minutes_per_shift
    for d in range(plan.moves.shape[0]):
        for j in range(problem.n):
            for i in range(problem.n):
                k = int(plan.moves[d, j, i])
                if i != j and k > 0:
                    t = int(problem.travel[j, i])
                    rows.append(
                        {
                            "day": d + 1,
                            "from_zone": problem.zone_ids[j],
                            "from_name": problem.zone_names[j],
                            "to_zone": problem.zone_ids[i],
                            "to_name": problem.zone_names[i],
                            "teams": k,
                            "travel_minutes_each": t,
                            "team_minutes": k * t,
                            "service_lost_requests": round(k * problem.service_per_team * t / shift, 2),
                        }
                    )
    cols = [
        "day",
        "from_zone",
        "from_name",
        "to_zone",
        "to_name",
        "teams",
        "travel_minutes_each",
        "team_minutes",
        "service_lost_requests",
    ]
    return pd.DataFrame(rows, columns=cols)


def day_allocation_matrix(problem: MultiPeriodProblem, plan: DailyPlan, d: int) -> np.ndarray:
    """Day d (0-based) as a single-period matrix: x[j, i] = moves j -> i, x[j, j] = teams that stayed.

    Column sums equal the teams stationed at the end of day d, so the single-period
    map and tables can show one day of a daily plan.
    """
    start = problem.available if d == 0 else plan.stationed[:, d - 1]
    x = plan.moves[d].astype(int).copy()
    np.fill_diagonal(x, np.asarray(start, dtype=int) - x.sum(axis=1))
    return x


def stationed_table(problem: MultiPeriodProblem, plan: DailyPlan) -> pd.DataFrame:
    """Zone x day matrix of stationed teams (day 0 = current positions)."""
    data = np.column_stack([problem.available, plan.stationed])
    cols = ["Day 0 (now)"] + [f"Day {d}" for d in range(1, data.shape[1])]
    return pd.DataFrame(
        data, index=[f"{z} {n}" for z, n in zip(problem.zone_ids, problem.zone_names, strict=True)], columns=cols
    )


def _build(problem: MultiPeriodProblem):
    n, H = problem.n, problem.horizon
    R, shift = int(problem.service_per_team), int(problem.minutes_per_shift)
    total = int(problem.available.sum())
    keep = problem.keep
    m = cp_model.CpModel()
    X = {(i, d): m.new_int_var(0, total, f"X_{i}_{d}") for i in range(n) for d in range(1, H + 1)}
    y = {
        (j, i, d): m.new_int_var(0, total, f"y_{j}_{i}_{d}")
        for d in range(1, H + 1)
        for j in range(n)
        for i in range(n)
        if i != j and problem.eligible[j, i]
    }

    def prev(i, d):
        return int(problem.available[i]) if d == 1 else X[i, d - 1]

    S, U = {}, {}
    for d in range(1, H + 1):
        for i in range(n):
            inflow = [y[j, i, d] for j in range(n) if (j, i, d) in y]
            outflow = [y[i, k, d] for k in range(n) if (i, k, d) in y]
            m.add(X[i, d] == prev(i, d) + sum(inflow) - sum(outflow))
            m.add(prev(i, d) - sum(outflow) >= int(keep[i]))
            m.add(X[i, d] <= int(problem.staging_capacity[i]))
            lo = int(problem.floor_requirement[i, d - 1]) if problem.coverage_floor > 0 else 0
            s = m.new_int_var(lo, int(problem.servable[i, d - 1]), f"S_{i}_{d}")
            lost = sum(R * int(problem.travel[j, i]) * y[j, i, d] for j in range(n) if (j, i, d) in y)
            m.add(shift * s <= R * shift * X[i, d] - lost)
            u = m.new_int_var(0, int(problem.demand[i, d - 1]), f"U_{i}_{d}")
            m.add(u == int(problem.demand[i, d - 1]) - s)
            S[i, d], U[i, d] = s, u
    w_int = [int(round(float(w) * WEIGHT_SCALE)) for w in problem.weights]
    weighted_unmet = sum(w_int[i] * U[i, d] for (i, d) in U)
    travel = sum(int(problem.travel[j, i]) * v for (j, i, _), v in y.items())
    return m, X, y, weighted_unmet, travel


def _extract(problem: MultiPeriodProblem, solver, X, y) -> DailyPlan:
    n, H = problem.n, problem.horizon
    stationed = np.zeros((n, H), dtype=int)
    moves = np.zeros((H, n, n), dtype=int)
    for (i, d), var in X.items():
        stationed[i, d - 1] = solver.value(var)
    for (j, i, d), var in y.items():
        moves[d - 1, j, i] = solver.value(var)
    return DailyPlan(stationed=stationed, moves=moves)


def _as_single_period(problem: MultiPeriodProblem) -> AllocationProblem:
    """Day-1 view used for infeasibility diagnosis (static constraints are identical)."""
    p = problem.day_problem(0, problem.available)
    return replace(p, coverage_floor=0.0)


def diagnose_daily_infeasibility(problem: MultiPeriodProblem) -> list[str]:
    reasons = diagnose_infeasibility(_as_single_period(problem))
    if problem.coverage_floor > 0:
        reasons.append(
            f"The {problem.coverage_floor:.0%} coverage floor applies on EVERY day; travel time also "
            "reduces a moving team's service on the day it moves, so the floor may be unreachable early "
            "in the horizon."
        )
    return reasons


def optimize_daily(
    problem: MultiPeriodProblem,
    mode: str = "lexicographic",
    travel_penalty_lambda: float = 0.01,
    time_limit_seconds: float = 10.0,
    num_workers: int = 1,
    seed: int = 0,
    hint: DailyPlan | None = None,
) -> DailyResult:
    """Solve the day-by-day problem.

    ``hint`` (e.g. the day-by-day greedy plan) warm-starts the search. On large
    problems the time limit can stop CP-SAT before it reaches a good solution on its
    own; starting from a valid plan lets it spend the time improving on it instead.
    """
    if mode not in {"lexicographic", "weighted"}:
        raise ValueError("mode must be 'lexicographic' or 'weighted'")
    start = time.perf_counter()
    model, X, y, weighted_unmet, travel = _build(problem)
    if hint is not None:
        for (i, d), var in X.items():
            model.add_hint(var, int(hint.stationed[i, d - 1]))
        for (j, i, d), var in y.items():
            model.add_hint(var, int(hint.moves[d - 1, j, i]))
    messages: list[str] = []
    stages: dict[str, str] = {}
    bound: float | None = None  # proven lower bound on weighted unmet demand (lexicographic stage 1)

    if mode == "weighted":
        lam = int(round(travel_penalty_lambda * WEIGHT_SCALE))
        if lam == 0 and travel_penalty_lambda > 0:
            messages.append("lambda below scaling resolution (0.001); travel cost ignored.")
        model.minimize(weighted_unmet + lam * travel)
        solver = _solver(time_limit_seconds, num_workers, seed)
        code = solver.solve(model)
        stages["weighted"] = _STATUS[code]
        status = _STATUS[code]
        plan = _extract(problem, solver, X, y) if code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    else:
        model.minimize(weighted_unmet)
        solver = _solver(time_limit_seconds, num_workers, seed)
        code1 = solver.solve(model)
        stages["1_weighted_unmet"] = _STATUS[code1]
        plan = None
        status = _STATUS[code1]
        if code1 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            plan = _extract(problem, solver, X, y)
            best = int(round(solver.objective_value))
            bound = float(solver.best_objective_bound) / WEIGHT_SCALE
            model.add(weighted_unmet <= best)
            model.clear_objective()
            model.minimize(travel)
            model.clear_hints()  # a variable may carry only one hint
            for (i, d), var in X.items():
                model.add_hint(var, int(plan.stationed[i, d - 1]))
            for (j, i, d), var in y.items():
                model.add_hint(var, int(plan.moves[d - 1, j, i]))
            solver2 = _solver(time_limit_seconds, num_workers, seed)
            code2 = solver2.solve(model)
            stages["2_travel_minutes"] = _STATUS[code2]
            if code2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                plan = _extract(problem, solver2, X, y)
                status = "OPTIMAL" if (code1 == cp_model.OPTIMAL and code2 == cp_model.OPTIMAL) else "FEASIBLE"
            else:
                status = "FEASIBLE"
                messages.append("Travel-minimisation stage found no solution in time; stage-1 plan kept.")

    result = DailyResult(
        strategy="optimized",
        status=status,
        plan=plan,
        runtime_seconds=time.perf_counter() - start,
        messages=messages,
        stage_statuses=stages,
    )
    if plan is None:
        if status == "INFEASIBLE":
            result.diagnostics["infeasibility_reasons"] = diagnose_daily_infeasibility(problem)
            messages.append("No day-by-day plan satisfies all hard constraints.")
        else:
            messages.append(f"Solver returned {status}: no plan was produced within the time limit.")
        return result
    outcome = compute_daily_outcome(problem, plan)
    result.objective = {"weighted_unmet": outcome.weighted_unmet, "travel_minutes": outcome.travel_minutes}
    if bound is not None:
        result.diagnostics["weighted_unmet_lower_bound"] = bound
        result.diagnostics["weighted_unmet_gap"] = max(outcome.weighted_unmet - bound, 0.0)
    if status == "FEASIBLE":
        messages.append("Solution is feasible but optimality was not proven within the time limit.")
    return result
