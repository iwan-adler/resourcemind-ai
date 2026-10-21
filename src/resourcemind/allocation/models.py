"""Allocation problem and result data structures plus the shared outcome calculation.

Index convention: ``x[j, i]`` = number of teams whose origin is zone j and that
are stationed in zone i for the planning horizon (``x[j, j]`` = teams staying).

All strategies (current, greedy, optimised) are scored with ``compute_outcome``
so they are compared on identical rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationProblem:
    zone_ids: tuple[str, ...]
    zone_names: tuple[str, ...]
    priority_tiers: tuple[str, ...]
    demand: np.ndarray  # D_i, integer service requests over the horizon
    available: np.ndarray  # A_j, integer teams currently at origin j
    staging_capacity: np.ndarray  # M_i, max teams that can be stationed in zone i
    service_cap: np.ndarray  # K_i, max useful service requests in zone i over the horizon
    weights: np.ndarray  # w_i, priority weight (dimensionless, >= 1 by default policy)
    travel: np.ndarray  # T[j, i], minutes from j to i (0 on the diagonal)
    eligible: np.ndarray  # bool[j, i], route allowed (diagonal always True)
    retain: np.ndarray  # minimum teams that must stay at their origin j
    service_per_team: int  # R, service requests one team can handle over the horizon
    horizon_days: int
    max_travel_minutes: int
    # Optional equity policy: each zone must receive at least this share of its
    # servable demand, ceil(floor * min(D_i, K_i)). 0 disables it.
    coverage_floor: float = 0.0
    # Optional response-time standard (allocation/response.py). None = in-zone model:
    # demand is served only by teams stationed in the same zone, and any response counts.
    # Otherwise response_pairs[k, i] = a team based in k reaches zone i within the standard.
    response_pairs: np.ndarray | None = None
    response_standard_minutes: float | None = None

    @property
    def n(self) -> int:
        return len(self.zone_ids)

    @property
    def uses_response_standard(self) -> bool:
        return self.response_pairs is not None

    @property
    def servable(self) -> np.ndarray:
        """Demand that could be served with unlimited teams: min(D_i, K_i)."""
        return np.minimum(self.demand, self.service_cap)

    @property
    def floor_requirement(self) -> np.ndarray:
        """Minimum requests that must be served per zone under the coverage floor."""
        return np.ceil(self.coverage_floor * self.servable - 1e-9).astype(int)


@dataclass
class AllocationResult:
    strategy: str
    status: str  # OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN / MODEL_INVALID / HEURISTIC / BASELINE
    x: np.ndarray | None  # teams matrix, None when no solution exists
    runtime_seconds: float
    messages: list[str] = field(default_factory=list)
    stage_statuses: dict[str, str] = field(default_factory=dict)
    objective: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_solution(self) -> bool:
        return self.x is not None


@dataclass
class Outcome:
    stationed: np.ndarray  # X_i
    served: np.ndarray  # S_i
    unmet: np.ndarray  # U_i
    weighted_unmet: float
    travel_minutes: float
    teams_moved: int
    limiting_factor: list[str]


def compute_outcome(problem: AllocationProblem, x: np.ndarray) -> Outcome:
    """Service delivered by an allocation.

    In-zone model: S_i = min(D_i, K_i, R * X_i). With a response-time standard, S is
    the best service teams can give within the standard from where they are stationed
    (an exact maximum-weight assignment, see allocation/response.py).
    """
    x = np.asarray(x)
    stationed = x.sum(axis=0)
    if problem.uses_response_standard:
        from resourcemind.allocation.response import best_service

        served, _ = best_service(
            stationed, problem.servable, problem.weights, problem.service_per_team, problem.response_pairs
        )
    else:
        served = np.minimum(problem.servable, problem.service_per_team * stationed)
    unmet = problem.demand - served
    off_diag = x * (1 - np.eye(problem.n, dtype=int))
    limiting = []
    for i in range(problem.n):
        if problem.demand[i] == 0 or served[i] >= problem.demand[i]:
            limiting.append("demand fully served")
        elif served[i] >= problem.service_cap[i]:
            limiting.append("service cap K (more teams would not help)")
        elif problem.uses_response_standard and not (problem.response_pairs[:, i] & (stationed > 0)).any():
            limiting.append("no team stationed within the response standard")
        elif stationed[i] >= problem.staging_capacity[i]:
            limiting.append("staging capacity M")
        else:
            limiting.append("team capacity (not enough teams)")
    return Outcome(
        stationed=stationed,
        served=served,
        unmet=unmet,
        weighted_unmet=float((problem.weights * unmet).sum()),
        travel_minutes=float((problem.travel * off_diag).sum()),
        teams_moved=int(off_diag.sum()),
        limiting_factor=limiting,
    )


def movements_table(problem: AllocationProblem, x: np.ndarray) -> pd.DataFrame:
    """Rows for each origin -> destination move with n_teams > 0 (excluding stays)."""
    rows = []
    for j in range(problem.n):
        for i in range(problem.n):
            if i != j and x[j, i] > 0:
                rows.append(
                    {
                        "from_zone": problem.zone_ids[j],
                        "from_name": problem.zone_names[j],
                        "to_zone": problem.zone_ids[i],
                        "to_name": problem.zone_names[i],
                        "teams": int(x[j, i]),
                        "travel_minutes_each": int(problem.travel[j, i]),
                        "team_minutes": int(x[j, i] * problem.travel[j, i]),
                    }
                )
    cols = ["from_zone", "from_name", "to_zone", "to_name", "teams", "travel_minutes_each", "team_minutes"]
    return pd.DataFrame(rows, columns=cols)
