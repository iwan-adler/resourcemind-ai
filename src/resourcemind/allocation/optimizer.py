"""Constrained team allocation with Google OR-Tools CP-SAT.

Decision variables (all integer):
    x[j, i] in [0, A_j]  teams from origin j stationed in zone i (only eligible routes)
    S_i in [0, min(D_i, K_i)]  service requests delivered in zone i
    U_i = D_i - S_i            unmet requests in zone i

Constraints:
    sum_i x[j, i] = A_j          every team is somewhere (conservation)
    x[j, j] >= min(retain_j, A_j) minimum local presence
    X_i = sum_j x[j, i] <= M_i   staging capacity
    S_i <= R * X_i               team service capacity
    (S_i <= D_i and S_i <= K_i via the variable domain)

Objectives (see docs/mathematical_model.md):
    lexicographic: (1) minimise sum_i w_i U_i, then (2) minimise sum_{j != i} T_ji x_ji
                   subject to stage-1 optimum
    weighted:      minimise sum_i w_i U_i + lambda * sum_{j != i} T_ji x_ji

CP-SAT needs integer coefficients, so weights are scaled by WEIGHT_SCALE.
"""

from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model

from resourcemind.allocation.models import AllocationProblem, AllocationResult, compute_outcome

WEIGHT_SCALE = 1000  # w_i = 1.37 -> 1370; resolution of 0.001 weighted requests

_STATUS = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


def _build(problem: AllocationProblem):
    n = problem.n
    m = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for j in range(n):
        for i in range(n):
            if problem.eligible[j, i] or i == j:
                x[j, i] = m.new_int_var(0, int(problem.available[j]), f"x_{j}_{i}")
    for j in range(n):
        m.add(sum(x[j, i] for i in range(n) if (j, i) in x) == int(problem.available[j]))
        keep = min(int(problem.retain[j]), int(problem.available[j]))
        if keep > 0:
            m.add(x[j, j] >= keep)
    S, U = [], []
    for i in range(n):
        incoming = [x[j, i] for j in range(n) if (j, i) in x]
        m.add(sum(incoming) <= int(problem.staging_capacity[i]))
        s = m.new_int_var(
            int(problem.floor_requirement[i]) if problem.coverage_floor > 0 else 0,
            int(problem.servable[i]),
            f"S_{i}",
        )
        m.add(s <= int(problem.service_per_team) * sum(incoming))
        u = m.new_int_var(0, int(problem.demand[i]), f"U_{i}")
        m.add(u == int(problem.demand[i]) - s)
        S.append(s)
        U.append(u)
    w_int = [int(round(float(w) * WEIGHT_SCALE)) for w in problem.weights]
    weighted_unmet = sum(w_int[i] * U[i] for i in range(n))
    travel = sum(int(problem.travel[j, i]) * v for (j, i), v in x.items() if i != j)
    return m, x, weighted_unmet, travel


def _solver(time_limit: float, num_workers: int, seed: int) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(num_workers)
    solver.parameters.random_seed = int(seed)
    # Full LP linearisation gives a much tighter bound for S_i <= R * X_i, so a
    # single deterministic worker proves optimality quickly (measured: one surge
    # instance went from "FEASIBLE after 30 s" to "OPTIMAL in 0.05 s").
    solver.parameters.linearization_level = 2
    return solver


def _extract(problem, solver, x) -> np.ndarray:
    out = np.zeros((problem.n, problem.n), dtype=int)
    for (j, i), var in x.items():
        out[j, i] = solver.value(var)
    return out


def diagnose_infeasibility(problem: AllocationProblem) -> list[str]:
    """Plain-language reasons why no allocation can satisfy all hard constraints."""
    reasons = []
    n = problem.n
    total, cap = int(problem.available.sum()), int(problem.staging_capacity.sum())
    if total > cap:
        reasons.append(f"{total} teams must be stationed somewhere but total staging capacity is {cap}.")
    for j in range(n):
        has_exit = any(problem.eligible[j, i] for i in range(n) if i != j)
        stuck = int(problem.available[j]) if not has_exit else min(int(problem.retain[j]), int(problem.available[j]))
        if stuck > problem.staging_capacity[j]:
            why = "it has no eligible outbound route" if not has_exit else "of the minimum-retention rule"
            reasons.append(
                f"{problem.zone_ids[j]} ({problem.zone_names[j]}) must keep {stuck} team(s) because {why}, "
                f"but its staging capacity is {problem.staging_capacity[j]}."
            )
    if problem.coverage_floor > 0:
        R = problem.service_per_team
        need = np.ceil(problem.floor_requirement / max(R, 1)).astype(int) if R > 0 else problem.floor_requirement
        if R <= 0 and problem.floor_requirement.sum() > 0:
            reasons.append("Coverage floor requires service but teams have zero service capacity.")
        elif need.sum() > total:
            reasons.append(
                f"A {problem.coverage_floor:.0%} coverage floor needs at least {int(need.sum())} teams "
                f"but only {total} are available."
            )
        else:
            reasons.append(
                f"The {problem.coverage_floor:.0%} coverage floor may be unreachable because some zones "
                "cannot receive enough teams over eligible routes or within staging capacity."
            )
    relax = staging_relaxation(problem) if problem.coverage_floor == 0 else {}
    if relax:
        extra = ", ".join(f"{z}: +{k}" for z, k in relax.items())
        reasons.append(f"Smallest staging-capacity increase that restores feasibility: {extra}.")
    if not reasons:
        reasons.append("Constraints conflict (conservation, retention, staging capacity and route eligibility).")
    return reasons


def staging_relaxation(problem: AllocationProblem, time_limit: float = 5.0) -> dict[str, int]:
    """Minimum extra staging slots per zone (elastic model) needed for feasibility."""
    n = problem.n
    m = cp_model.CpModel()
    x = {}
    for j in range(n):
        for i in range(n):
            if problem.eligible[j, i] or i == j:
                x[j, i] = m.new_int_var(0, int(problem.available[j]), f"x_{j}_{i}")
    extra = [m.new_int_var(0, int(problem.available.sum()), f"e_{i}") for i in range(n)]
    for j in range(n):
        m.add(sum(x[j, i] for i in range(n) if (j, i) in x) == int(problem.available[j]))
        keep = min(int(problem.retain[j]), int(problem.available[j]))
        if keep > 0:
            m.add(x[j, j] >= keep)
    for i in range(n):
        m.add(sum(x[j, i] for j in range(n) if (j, i) in x) <= int(problem.staging_capacity[i]) + extra[i])
    m.minimize(sum(extra))
    solver = _solver(time_limit, 1, 0)
    if solver.solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}
    return {problem.zone_ids[i]: int(solver.value(e)) for i, e in enumerate(extra) if solver.value(e) > 0}


def optimize_allocation(
    problem: AllocationProblem,
    mode: str = "lexicographic",
    travel_penalty_lambda: float = 0.01,
    time_limit_seconds: float = 10.0,
    num_workers: int = 1,
    seed: int = 0,
) -> AllocationResult:
    if mode not in {"lexicographic", "weighted"}:
        raise ValueError("mode must be 'lexicographic' or 'weighted'")
    start = time.perf_counter()
    model, x, weighted_unmet, travel = _build(problem)
    messages: list[str] = []
    stages: dict[str, str] = {}

    if mode == "weighted":
        lam = int(round(travel_penalty_lambda * WEIGHT_SCALE))
        if lam == 0 and travel_penalty_lambda > 0:
            messages.append("lambda below scaling resolution (0.001); travel cost ignored.")
        model.minimize(weighted_unmet + lam * travel)
        solver = _solver(time_limit_seconds, num_workers, seed)
        code = solver.solve(model)
        stages["weighted"] = _STATUS[code]
        status = _STATUS[code]
        solution = _extract(problem, solver, x) if code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    else:
        model.minimize(weighted_unmet)
        solver = _solver(time_limit_seconds, num_workers, seed)
        code1 = solver.solve(model)
        stages["1_weighted_unmet"] = _STATUS[code1]
        solution = None
        status = _STATUS[code1]
        if code1 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = _extract(problem, solver, x)
            best = int(round(solver.objective_value))
            # Stage 2: keep stage-1 performance, minimise travel.
            model.add(weighted_unmet <= best)
            model.clear_objective()
            model.minimize(travel)
            for (j, i), var in x.items():
                model.add_hint(var, int(solution[j, i]))
            solver2 = _solver(time_limit_seconds, num_workers, seed)
            code2 = solver2.solve(model)
            stages["2_travel_minutes"] = _STATUS[code2]
            if code2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                solution = _extract(problem, solver2, x)
                status = "OPTIMAL" if (code1 == cp_model.OPTIMAL and code2 == cp_model.OPTIMAL) else "FEASIBLE"
            else:
                status = "FEASIBLE"
                messages.append("Travel-minimisation stage found no solution in time; stage-1 plan kept.")

    runtime = time.perf_counter() - start
    result = AllocationResult(
        strategy="optimized",
        status=status,
        x=solution,
        runtime_seconds=runtime,
        messages=messages,
        stage_statuses=stages,
    )
    if solution is None:
        if status == "INFEASIBLE":
            result.diagnostics["infeasibility_reasons"] = diagnose_infeasibility(problem)
            messages.append("No allocation satisfies all hard constraints.")
        else:
            messages.append(f"Solver returned {status}: no allocation was produced within the time limit.")
        return result

    outcome = compute_outcome(problem, solution)
    result.objective = {
        "weighted_unmet": outcome.weighted_unmet,
        "travel_minutes": outcome.travel_minutes,
    }
    if status == "FEASIBLE":
        messages.append("Solution is feasible but optimality was not proven within the time limit.")
    return result
