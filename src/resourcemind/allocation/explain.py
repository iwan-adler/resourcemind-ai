"""Plain-language explanations derived from the computed plan (no LLM involved)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from resourcemind.allocation.models import AllocationProblem, Outcome


def zone_rationale(problem: AllocationProblem, x: np.ndarray, before: Outcome, after: Outcome) -> pd.DataFrame:
    """One row per zone explaining why it gained, lost or kept teams."""
    rows = []
    R = problem.service_per_team
    for i in range(problem.n):
        delta = int(after.stationed[i] - before.stationed[i])
        demand, cap = int(problem.demand[i]), int(problem.service_cap[i])
        # The diagonal (staying put) is always eligible, so a sum of 1 means no other route.
        isolated = problem.eligible[:, i].sum() == 1 and problem.eligible[i, :].sum() == 1
        access = "isolated: no eligible routes in or out" if isolated else ""
        if delta > 0:
            reason = (
                f"Receives {delta} team(s): forecast {demand} requests exceeds its current capacity "
                f"({int(before.stationed[i]) * R}); priority weight {problem.weights[i]:.2f}."
            )
        elif delta < 0:
            reason = (
                f"Releases {-delta} team(s): remaining {int(after.stationed[i])} team(s) can still serve "
                f"{int(after.served[i])} of {demand} requests; the released capacity serves more "
                "weighted demand elsewhere."
            )
        else:
            if before.unmet[i] == 0:
                reason = "Keeps its teams: demand is fully covered and no surplus is worth moving."
            elif before.served[i] >= cap:
                reason = f"Keeps its teams: service cap K={cap} is reached, extra teams would not help."
            elif after.stationed[i] >= problem.staging_capacity[i]:
                reason = "Keeps its teams: staging capacity is full."
            else:
                reason = (
                    "Keeps its teams: no eligible spare team elsewhere can serve more weighted demand here "
                    "than where it currently is."
                )
        if access:
            reason += f" ({access})"
        rows.append(
            {
                "zone_id": problem.zone_ids[i],
                "zone_name": problem.zone_names[i],
                "teams_before": int(before.stationed[i]),
                "teams_after": int(after.stationed[i]),
                "change": delta,
                "unmet_before": int(before.unmet[i]),
                "unmet_after": int(after.unmet[i]),
                "limiting_factor": after.limiting_factor[i],
                "rationale": reason,
            }
        )
    return pd.DataFrame(rows)


def plan_summary(problem: AllocationProblem, before: Outcome, after: Outcome) -> list[str]:
    total_demand = int(problem.demand.sum())
    unservable = int((problem.demand - problem.servable).sum())
    lines = [
        f"Moving {after.teams_moved} team(s) ({after.travel_minutes:.0f} team-minutes of travel) changes "
        f"weighted unmet demand from {before.weighted_unmet:.1f} to {after.weighted_unmet:.1f}.",
        f"Requests served: {int(before.served.sum())} -> {int(after.served.sum())} of {total_demand}.",
    ]
    if unservable > 0:
        lines.append(f"{unservable} requests exceed zone service caps (K) and cannot be served by any allocation.")
    total_capacity = int(problem.available.sum()) * problem.service_per_team
    if total_capacity < problem.servable.sum():
        lines.append(
            f"System-wide shortage: total team capacity {total_capacity} < servable demand "
            f"{int(problem.servable.sum())}; unmet demand is unavoidable."
        )
    return lines
