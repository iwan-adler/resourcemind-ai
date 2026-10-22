"""Baseline strategies that use exactly the same AllocationProblem as the optimiser."""

from __future__ import annotations

import time

import numpy as np

from resourcemind.allocation.models import AllocationProblem, AllocationResult
from resourcemind.allocation.multiperiod import DailyPlan, DailyResult, MultiPeriodProblem


def current_allocation(problem: AllocationProblem) -> AllocationResult:
    """Keep every team where it is today: x[j, j] = A_j."""
    start = time.perf_counter()
    x = np.diag(problem.available.astype(int))
    return AllocationResult(
        strategy="current",
        status="BASELINE",
        x=x,
        runtime_seconds=time.perf_counter() - start,
        messages=["No teams moved (status quo)."],
    )


def _served(problem: AllocationProblem, i: int, teams: int) -> int:
    return int(min(problem.servable[i], problem.service_per_team * max(teams, 0)))


def greedy_allocation(problem: AllocationProblem) -> AllocationResult:
    """Transparent single-team moves, best first.

    Repeatedly move one team that has not yet moved, from origin j to zone i,
    choosing the move with the highest score

        score = (gain_i - loss_j) / (1 + T_ji / 60)

    where gain_i = w_i * (extra requests served in i by one more team) and
    loss_j = w_j * (requests no longer served in j). Only moves with positive
    net gain on an eligible route, below staging capacity and respecting the
    retention minimum are considered. Stops when no move improves the score.
    Ties break on lower travel time, then lower zone indices (deterministic).
    A repair phase first tries to move teams out of zones above staging capacity.

    The rule always reasons in-zone (a team serves its own zone). Under a response-time
    standard it is therefore a simpler heuristic than the model it is scored with: its
    plan is still scored by the same response-aware outcome function as the optimizer.
    """
    start = time.perf_counter()
    n = problem.n
    x = np.diag(problem.available.astype(int))
    stationed = x.sum(axis=0)
    keep = np.minimum(problem.retain, problem.available).astype(int)
    w = problem.weights
    messages = []

    def candidate_moves(origins, require_positive: bool):
        best = None
        for j in origins:
            if x[j, j] <= keep[j]:
                continue
            loss = w[j] * (_served(problem, j, stationed[j]) - _served(problem, j, stationed[j] - 1))
            for i in range(n):
                if i == j or not problem.eligible[j, i] or stationed[i] >= problem.staging_capacity[i]:
                    continue
                gain = w[i] * (_served(problem, i, stationed[i] + 1) - _served(problem, i, stationed[i]))
                net = gain - loss
                if require_positive and net <= 1e-12:
                    continue
                score = net / (1.0 + problem.travel[j, i] / 60.0)
                key = (score, -problem.travel[j, i], -j, -i)
                if best is None or key > best[0]:
                    best = (key, j, i)
        return best

    # Repair phase: zones over staging capacity push teams out if they can.
    for j in range(n):
        while stationed[j] > problem.staging_capacity[j]:
            move = candidate_moves([j], require_positive=False)
            if move is None:
                messages.append(f"{problem.zone_ids[j]} remains above staging capacity (no eligible move).")
                break
            _, jj, ii = move
            x[jj, jj] -= 1
            x[jj, ii] += 1
            stationed[jj] -= 1
            stationed[ii] += 1

    moves = 0
    while True:
        move = candidate_moves(range(n), require_positive=True)
        if move is None:
            break
        _, j, i = move
        x[j, j] -= 1
        x[j, i] += 1
        stationed[j] -= 1
        stationed[i] += 1
        moves += 1
    messages.append(f"{moves} single-team move(s) made by the greedy rule.")
    return AllocationResult(
        strategy="greedy",
        status="HEURISTIC",
        x=x,
        runtime_seconds=time.perf_counter() - start,
        messages=messages,
    )


def current_daily(problem: MultiPeriodProblem) -> DailyResult:
    """Keep every team where it is today, every day."""
    start = time.perf_counter()
    plan = DailyPlan.from_moves(problem.available, np.zeros((problem.horizon, problem.n, problem.n), dtype=int))
    return DailyResult(
        strategy="current",
        status="BASELINE",
        plan=plan,
        runtime_seconds=time.perf_counter() - start,
        messages=["No teams moved on any day (status quo)."],
    )


def greedy_daily(problem: MultiPeriodProblem) -> DailyResult:
    """The single-period greedy rule applied myopically, one day at a time.

    Day d starts from the positions at the end of day d-1 and runs
    ``greedy_allocation`` against that day's demand only (no look-ahead). The
    greedy rule does not see the service lost while travelling; the shared
    outcome function charges it afterwards, exactly as for the optimizer.
    """
    start = time.perf_counter()
    n, H = problem.n, problem.horizon
    moves = np.zeros((H, n, n), dtype=int)
    positions = problem.available.astype(int)
    messages = []
    for d in range(H):
        day = greedy_allocation(problem.day_problem(d, positions))
        moves[d] = day.x * (1 - np.eye(n, dtype=int))
        positions = day.x.sum(axis=0)
        messages += [f"Day {d + 1}: {m}" for m in day.messages if "above staging capacity" in m]
    plan = DailyPlan.from_moves(problem.available, moves)
    messages.append(f"{int(moves.sum())} single-team move(s) over {H} day(s) by the day-by-day greedy rule.")
    return DailyResult(
        strategy="greedy",
        status="HEURISTIC",
        plan=plan,
        runtime_seconds=time.perf_counter() - start,
        messages=messages,
    )
