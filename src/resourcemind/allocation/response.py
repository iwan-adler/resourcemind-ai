"""Service within a response-time standard, given where teams are stationed.

With a response-time standard tau, demand in zone i counts as covered only when it
is served by a team that can reach it within tau. ``pairs[k, i]`` is True when a team
based in zone k can do that (k == i: the zone's local response time is within tau;
k != i: the travel time from k to i is within tau). Each team's capacity R is
shared across the zones it can reach in time:

    sum_i f[k, i] <= R * X_k      (team capacity at base k)
    sum_k f[k, i] <= min(D_i, K_i) (useful demand in zone i)
    S_i = sum_k f[k, i]

Given positions X, the best service is the maximum-weight flow, solved exactly with
CP-SAT. The default in-zone model is the special case pairs = identity, which has
the closed form S_i = min(D_i, K_i, R * X_i).
"""

from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

WEIGHT_SCALE = 1000


def best_service(
    stationed: np.ndarray,
    servable: np.ndarray,
    weights: np.ndarray,
    rate: int,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum weighted service S_i and flows f[k, i] for fixed team positions."""
    n = len(servable)
    stationed = np.asarray(stationed, dtype=int)
    servable = np.asarray(servable, dtype=int)
    pairs = np.asarray(pairs, dtype=bool)
    flows = np.zeros((n, n), dtype=int)
    if not (pairs & ~np.eye(n, dtype=bool)).any():  # in-zone only: closed form
        diag = np.diag(pairs)
        served = np.where(diag, np.minimum(servable, rate * stationed), 0)
        np.fill_diagonal(flows, served)
        return served.astype(int), flows
    m = cp_model.CpModel()
    f = {}
    for k in range(n):
        if stationed[k] <= 0:
            continue
        for i in range(n):
            if pairs[k, i] and servable[i] > 0:
                f[k, i] = m.new_int_var(0, int(min(servable[i], rate * stationed[k])), f"f_{k}_{i}")
    for k in range(n):
        out = [v for (kk, _), v in f.items() if kk == k]
        if out:
            m.add(sum(out) <= int(rate * stationed[k]))
    for i in range(n):
        inc = [v for (_, ii), v in f.items() if ii == i]
        if inc:
            m.add(sum(inc) <= int(servable[i]))
    w = [int(round(float(x) * WEIGHT_SCALE)) for x in weights]
    m.maximize(sum(w[i] * v for (_, i), v in f.items()))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.solve(m)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"service assignment not solved to optimality ({solver.status_name(status)})")
    for (k, i), v in f.items():
        flows[k, i] = solver.value(v)
    return flows.sum(axis=0).astype(int), flows


def response_pairs(travel: np.ndarray, local_response: np.ndarray, standard: float, blocked: np.ndarray) -> np.ndarray:
    """pairs[k, i]: a team based in k reaches zone i within ``standard`` minutes.

    ``blocked[k, i]`` marks routes that cannot be used (blocked roads, inaccessible
    zones); a team inside an inaccessible zone can still serve its own zone.
    """
    pairs = (np.asarray(travel, dtype=float) <= standard) & ~np.asarray(blocked, dtype=bool)
    np.fill_diagonal(pairs, np.asarray(local_response, dtype=float) <= standard)
    return pairs
