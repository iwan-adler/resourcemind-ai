"""Independent validation of any allocation matrix or day-by-day plan.

Every hard constraint is re-checked from the problem data rather than taken from
the solver. Optimizer and baseline outputs go through the same checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resourcemind.allocation.models import AllocationProblem, compute_outcome
from resourcemind.allocation.multiperiod import DailyPlan, MultiPeriodProblem, compute_daily_outcome


@dataclass(frozen=True)
class Violation:
    constraint: str
    zone_id: str
    detail: str


def validate_allocation(problem: AllocationProblem, x: np.ndarray | None) -> list[Violation]:
    if x is None:
        return []
    out: list[Violation] = []
    x = np.asarray(x)
    n = problem.n
    if x.shape != (n, n):
        return [Violation("shape", "-", f"expected {(n, n)} matrix, got {x.shape}")]
    if not np.issubdtype(x.dtype, np.integer) and not np.allclose(x, np.round(x)):
        out.append(Violation("integrality", "-", "team counts must be integers"))
    if (x < 0).any():
        out.append(Violation("non-negativity", "-", "negative team counts"))

    for j in range(n):
        zid = problem.zone_ids[j]
        sent = int(x[j].sum())
        if sent != int(problem.available[j]):
            out.append(
                Violation(
                    "resource conservation",
                    zid,
                    f"origin has {problem.available[j]} teams but {sent} are assigned",
                )
            )
        keep = min(int(problem.retain[j]), int(problem.available[j]))
        if x[j, j] < keep:
            out.append(Violation("minimum retention", zid, f"{x[j, j]} teams stay, minimum is {keep}"))
        for i in range(n):
            if x[j, i] > 0 and not problem.eligible[j, i]:
                reason = (
                    f"travel {problem.travel[j, i]} min > limit {problem.max_travel_minutes} min"
                    if problem.travel[j, i] > problem.max_travel_minutes
                    else "route blocked or zone inaccessible"
                )
                out.append(
                    Violation("route eligibility", zid, f"{x[j, i]} team(s) {zid}->{problem.zone_ids[i]}: {reason}")
                )

    stationed = x.sum(axis=0)
    for i in range(n):
        if stationed[i] > problem.staging_capacity[i]:
            out.append(
                Violation(
                    "staging capacity",
                    problem.zone_ids[i],
                    f"{stationed[i]} teams stationed, capacity {problem.staging_capacity[i]}",
                )
            )
    if problem.coverage_floor > 0:
        served = compute_outcome(problem, x).served
        need = problem.floor_requirement
        for i in range(n):
            if served[i] < need[i]:
                out.append(
                    Violation(
                        "coverage floor",
                        problem.zone_ids[i],
                        f"serves {served[i]} requests, policy floor requires {need[i]}",
                    )
                )
    return out


def validate_daily_plan(problem: MultiPeriodProblem, plan: DailyPlan | None) -> list[Violation]:
    """Re-check every hard constraint of a day-by-day plan from the raw problem data."""
    if plan is None:
        return []
    out: list[Violation] = []
    n, H = problem.n, problem.horizon
    X = np.asarray(plan.stationed)
    y = np.asarray(plan.moves)
    if X.shape != (n, H) or y.shape != (H, n, n):
        return [Violation("shape", "-", f"expected stationed {(n, H)} and moves {(H, n, n)}")]
    for name, arr in (("stationed", X), ("moves", y)):
        if not np.issubdtype(arr.dtype, np.integer) and not np.allclose(arr, np.round(arr)):
            out.append(Violation("integrality", "-", f"{name} team counts must be integers"))
        if (arr < 0).any():
            out.append(Violation("non-negativity", "-", f"negative {name} team counts"))
    if (np.einsum("dii->di", y) != 0).any():
        out.append(Violation("moves", "-", "a zone cannot send teams to itself"))
    keep = np.minimum(problem.retain, problem.available)
    prev = np.asarray(problem.available, dtype=int)
    for d in range(H):
        day = f"day {d + 1}"
        inflow, outflow = y[d].sum(axis=0), y[d].sum(axis=1)
        for i in range(n):
            zid = problem.zone_ids[i]
            expected = int(prev[i] + inflow[i] - outflow[i])
            if X[i, d] != expected:
                out.append(
                    Violation(
                        "resource conservation",
                        zid,
                        f"{day}: {X[i, d]} teams stationed but {prev[i]} + {inflow[i]} in - {outflow[i]} out "
                        f"= {expected}",
                    )
                )
            if prev[i] - outflow[i] < keep[i]:
                out.append(
                    Violation(
                        "minimum retention", zid, f"{day}: {prev[i] - outflow[i]} teams stay, minimum is {keep[i]}"
                    )
                )
            if X[i, d] > problem.staging_capacity[i]:
                out.append(
                    Violation(
                        "staging capacity",
                        zid,
                        f"{day}: {X[i, d]} teams stationed, capacity {problem.staging_capacity[i]}",
                    )
                )
            for k in range(n):
                if k != i and y[d, i, k] > 0 and not problem.eligible[i, k]:
                    reason = (
                        f"travel {problem.travel[i, k]} min > limit {problem.max_travel_minutes} min"
                        if problem.travel[i, k] > problem.max_travel_minutes
                        else "route blocked or zone inaccessible"
                    )
                    out.append(
                        Violation(
                            "route eligibility",
                            zid,
                            f"{day}: {y[d, i, k]} team(s) {zid}->{problem.zone_ids[k]}: {reason}",
                        )
                    )
        if X[:, d].sum() != problem.available.sum():
            out.append(
                Violation(
                    "resource conservation", "-", f"{day}: {X[:, d].sum()} teams, {problem.available.sum()} exist"
                )
            )
        # The next day starts from the claimed positions so one error is reported once.
        prev = X[:, d].astype(int)
    if problem.coverage_floor > 0:
        served = compute_daily_outcome(problem, plan).served
        need = problem.floor_requirement
        for i, d in zip(*np.nonzero(served < need), strict=True):
            out.append(
                Violation(
                    "coverage floor",
                    problem.zone_ids[i],
                    f"day {d + 1}: serves {served[i, d]} requests, policy floor requires {need[i, d]}",
                )
            )
    return out
