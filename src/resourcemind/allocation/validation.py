"""Independent validation of any allocation matrix.

This module does NOT trust the solver: it re-checks every hard constraint from
the raw problem data. Optimiser and baseline outputs pass through the same checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resourcemind.allocation.models import AllocationProblem, compute_outcome


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
