"""What-if analysis: re-solve the allocation model with one move forced or forbidden.

``force_move`` requires at least k teams on the route A -> B; ``forbid_move`` disallows it.
Both add one bound on x[A, B] to the lexicographic CP-SAT model and compare the result
with the recommended plan using ``compute_outcome``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from resourcemind.allocation.models import AllocationProblem, compute_outcome
from resourcemind.allocation.optimizer import optimize_allocation
from resourcemind.allocation.validation import validate_allocation


@dataclass
class Counterfactual:
    question: str
    feasible: bool
    status: str
    answer: str  # one plain-language sentence
    delta_weighted_unmet: float | None = None  # counterfactual - recommended (> 0 = worse)
    delta_served: float | None = None
    delta_travel: float | None = None
    x: np.ndarray | None = None
    zone_changes: pd.DataFrame = field(default_factory=pd.DataFrame)


def _route_problem(problem: AllocationProblem, j: int, i: int) -> str | None:
    if i == j:
        return "origin and destination are the same zone"
    if problem.eligible[j, i]:
        return None
    if problem.travel[j, i] > problem.max_travel_minutes:
        return (
            f"the trip takes {int(problem.travel[j, i])} min, over the {problem.max_travel_minutes}-min "
            "repositioning limit"
        )
    return "the route is blocked or one of the zones is inaccessible"


def _compare(problem, base_x, new_x) -> tuple[float, float, float, pd.DataFrame]:
    a, b = compute_outcome(problem, base_x), compute_outcome(problem, new_x)
    changes = pd.DataFrame(
        {
            "zone_id": problem.zone_ids,
            "zone_name": problem.zone_names,
            "teams_recommended": a.stationed,
            "teams_counterfactual": b.stationed,
            "served_recommended": a.served,
            "served_counterfactual": b.served,
            "served_change": b.served - a.served,
        }
    )
    changes = changes[
        (changes["teams_recommended"] != changes["teams_counterfactual"]) | (changes["served_change"] != 0)
    ]
    return (
        b.weighted_unmet - a.weighted_unmet,
        float(b.served.sum() - a.served.sum()),
        b.travel_minutes - a.travel_minutes,
        changes.reset_index(drop=True),
    )


def _solve(problem, bounds, config) -> tuple[str, np.ndarray | None]:
    opt = config.optimizer
    res = optimize_allocation(
        problem,
        mode=opt.mode,
        travel_penalty_lambda=opt.travel_penalty_lambda,
        time_limit_seconds=opt.time_limit_seconds,
        num_workers=opt.num_workers,
        seed=config.seed,
        move_bounds=bounds,
    )
    if res.x is not None and validate_allocation(problem, res.x):
        return "INVALID", None
    return res.status, res.x


def _describe(name_from, name_to, d_wu, d_served, d_travel) -> str:
    if abs(d_wu) < 1e-9 and abs(d_served) < 1e-9:
        cost = "it would serve exactly as well"
        if d_travel > 0:
            cost += f" but add {d_travel:.0f} team-minutes of travel"
        elif d_travel < 0:
            cost += f" and save {-d_travel:.0f} team-minutes of travel"
        return cost + "."
    direction = "worse" if d_wu > 0 else "better"
    return (
        f"weighted unmet demand would be {abs(d_wu):.1f} {direction} ({d_served:+.0f} requests served"
        f", {d_travel:+.0f} team-minutes of travel)."
    )


def force_move(
    problem: AllocationProblem, recommended_x: np.ndarray, j: int, i: int, teams: int, config
) -> Counterfactual:
    """What if at least ``teams`` teams moved from zone j to zone i? (everything else re-optimised)"""
    a, b = problem.zone_names[j], problem.zone_names[i]
    question = f"Why not move {teams} team(s) from {a} to {b}?"
    reason = _route_problem(problem, j, i)
    if reason:
        return Counterfactual(question, False, "NOT_ALLOWED", f"That move is not allowed: {reason}.")
    keep = min(int(problem.retain[j]), int(problem.available[j]))
    if teams > int(problem.available[j]) - keep:
        return Counterfactual(
            question,
            False,
            "NOT_ALLOWED",
            f"{a} has {int(problem.available[j])} team(s) and must keep {keep}, so at most "
            f"{max(int(problem.available[j]) - keep, 0)} can leave.",
        )
    status, x = _solve(problem, {(j, i): (teams, int(problem.available[j]))}, config)
    if x is None:
        return Counterfactual(question, False, status, f"No valid plan includes that move ({status}).")
    d_wu, d_served, d_travel, changes = _compare(problem, recommended_x, x)
    answer = "With that move forced and every other team re-optimised, " + _describe(a, b, d_wu, d_served, d_travel)
    return Counterfactual(question, True, status, answer, d_wu, d_served, d_travel, x, changes)


def forbid_move(problem: AllocationProblem, recommended_x: np.ndarray, j: int, i: int, config) -> Counterfactual:
    """What if no team could move from zone j to zone i? (everything else re-optimised)"""
    a, b = problem.zone_names[j], problem.zone_names[i]
    question = f"Why move teams from {a} to {b}?"
    status, x = _solve(problem, {(j, i): (0, 0)}, config)
    if x is None:
        return Counterfactual(
            question, False, status, f"Without that move no valid plan exists ({status}): it is required."
        )
    d_wu, d_served, d_travel, changes = _compare(problem, recommended_x, x)
    if abs(d_wu) < 1e-9 and abs(d_served) < 1e-9:
        answer = (
            "Without it, an equally good plan exists"
            + (f" with {d_travel:+.0f} team-minutes of travel" if d_travel else "")
            + "; the move is the cheapest of equivalent options, not essential."
        )
    else:
        answer = "Without it, " + _describe(a, b, d_wu, d_served, d_travel)
    return Counterfactual(question, True, status, answer, d_wu, d_served, d_travel, x, changes)


def recommended_moves(problem: AllocationProblem, x: np.ndarray) -> list[tuple[int, int, int]]:
    """(origin, destination, teams) for every off-diagonal move in a plan."""
    return [(j, i, int(x[j, i])) for j in range(problem.n) for i in range(problem.n) if i != j and x[j, i] > 0]


def value_of_moves(problem: AllocationProblem, x: np.ndarray, config) -> pd.DataFrame:
    """For each recommended move, what forbidding it would cost (one re-solve per move)."""
    rows = []
    for j, i, k in recommended_moves(problem, x):
        cf = forbid_move(problem, x, j, i, config)
        rows.append(
            {
                "from_zone": problem.zone_ids[j],
                "from_name": problem.zone_names[j],
                "to_zone": problem.zone_ids[i],
                "to_name": problem.zone_names[i],
                "teams": k,
                "weighted_unmet_if_forbidden": cf.delta_weighted_unmet,
                "requests_lost_if_forbidden": None if cf.delta_served is None else 0.0 - cf.delta_served + 0.0,
                "travel_change_if_forbidden": cf.delta_travel,
                "explanation": cf.answer,
            }
        )
    cols = [
        "from_zone",
        "from_name",
        "to_zone",
        "to_name",
        "teams",
        "weighted_unmet_if_forbidden",
        "requests_lost_if_forbidden",
        "travel_change_if_forbidden",
        "explanation",
    ]
    return pd.DataFrame(rows, columns=cols)
