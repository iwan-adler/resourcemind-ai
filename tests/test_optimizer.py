import numpy as np

from resourcemind.allocation.models import compute_outcome
from resourcemind.allocation.optimizer import diagnose_infeasibility, optimize_allocation
from resourcemind.allocation.validation import validate_allocation
from tests.conftest import make_problem


def _brute_force_best(problem):
    """Exhaustive search over all allocations of a tiny problem (ground truth)."""
    n = problem.n
    best = None

    def rows_for(j):
        a = int(problem.available[j])

        def rec(i, left, row):
            if i == n - 1:
                yield [*row, left]
                return
            for k in range(left + 1):
                yield from rec(i + 1, left - k, [*row, k])

        yield from rec(0, a, [])

    def search(j, x):
        nonlocal best
        if j == n:
            xm = np.array(x, dtype=int)
            if validate_allocation(problem, xm):
                return
            o = compute_outcome(problem, xm)
            key = (round(o.weighted_unmet, 6), o.travel_minutes)
            if best is None or key < best:
                best = key
            return
        for row in rows_for(j):
            search(j + 1, [*x, row])

    search(0, [])
    return best


def test_optimizer_moves_surplus_to_shortage(small_problem):
    res = optimize_allocation(small_problem)
    assert res.status == "OPTIMAL"
    assert validate_allocation(small_problem, res.x) == []
    o = compute_outcome(small_problem, res.x)
    # demand 100/20/60 with rate 20 and 6 teams: best is 5/1/... limited -> check optimum by brute force
    assert (round(o.weighted_unmet, 6), o.travel_minutes) == _brute_force_best(small_problem)


def test_matches_brute_force_on_varied_small_cases():
    cases = [
        make_problem(demand=(50, 0, 30), available=(0, 3, 1), weights=(2.0, 1.0, 1.0)),
        make_problem(demand=(40, 40, 40), available=(3, 0, 0), staging=(1, 2, 2), retain=1),
        make_problem(demand=(90, 10, 10), available=(1, 1, 2), service_cap=(45, 100, 100)),
        make_problem(
            demand=(30, 30, 30),
            available=(0, 2, 2),
            weights=(1.0, 1.5, 1.2),
            eligible=[[1, 1, 0], [1, 1, 0], [0, 0, 1]],
        ),
    ]
    for p in cases:
        res = optimize_allocation(p)
        assert res.status == "OPTIMAL"
        o = compute_outcome(p, res.x)
        assert (round(o.weighted_unmet, 6), o.travel_minutes) == _brute_force_best(p)


def test_resource_conservation_and_integrality(small_problem):
    res = optimize_allocation(small_problem)
    assert np.issubdtype(res.x.dtype, np.integer)
    np.testing.assert_array_equal(res.x.sum(axis=1), small_problem.available)
    assert (res.x >= 0).all()


def test_staging_and_service_caps_respected():
    p = make_problem(demand=(200, 0, 0), available=(0, 5, 5), staging=(3, 5, 7), service_cap=(50, 10, 10))
    res = optimize_allocation(p)
    o = compute_outcome(p, res.x)
    assert (o.stationed <= p.staging_capacity).all()
    assert (o.served <= p.service_cap).all()
    assert (o.served <= p.demand).all()
    # K=50 with rate 20 means 3 teams are enough; extra teams would add nothing
    assert o.stationed[0] == 3
    assert o.served[0] == 50


def test_shortage_calculation():
    p = make_problem(demand=(100, 0, 0), available=(2, 0, 0), rate=20)
    res = optimize_allocation(p)
    o = compute_outcome(p, res.x)
    assert o.served[0] == 40 and o.unmet[0] == 60


def test_blocked_routes_never_used():
    elig = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    p = make_problem(demand=(100, 0, 0), available=(0, 4, 4), eligible=elig)
    res = optimize_allocation(p)
    assert res.status == "OPTIMAL"
    np.testing.assert_array_equal(res.x, np.diag(p.available))


def test_travel_limit_respected():
    p = make_problem(demand=(100, 0, 0), available=(0, 0, 4), max_travel=15)  # 3->1 takes 20 min
    res = optimize_allocation(p)
    assert res.x[2, 0] == 0
    assert validate_allocation(p, res.x) == []


def test_infeasible_staging_reported_honestly():
    p = make_problem(available=(3, 3, 3), staging=(2, 2, 2))
    res = optimize_allocation(p)
    assert res.status == "INFEASIBLE"
    assert res.x is None
    reasons = res.diagnostics["infeasibility_reasons"]
    assert any("staging capacity" in r for r in reasons)


def test_infeasible_isolated_zone_explained():
    elig = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    p = make_problem(available=(4, 0, 0), staging=(2, 5, 5), eligible=elig)
    reasons = diagnose_infeasibility(p)
    assert any("no eligible outbound route" in r for r in reasons)
    assert optimize_allocation(p).status == "INFEASIBLE"


def test_zero_resources_and_zero_demand():
    p0 = make_problem(available=(0, 0, 0))
    r0 = optimize_allocation(p0)
    assert r0.status == "OPTIMAL"
    o0 = compute_outcome(p0, r0.x)
    assert o0.served.sum() == 0 and o0.unmet.sum() == p0.demand.sum()

    pd0 = make_problem(demand=(0, 0, 0))
    rd = optimize_allocation(pd0)
    assert rd.status == "OPTIMAL"
    assert compute_outcome(pd0, rd.x).teams_moved == 0  # nothing to gain from moving


def test_lexicographic_prefers_cheaper_equivalent_plan():
    # Zone 1 needs 1 team; both zone 2 (10 min) and zone 3 (20 min) can send it.
    p = make_problem(demand=(20, 0, 0), available=(0, 1, 1), retain=0)
    res = optimize_allocation(p)
    assert res.x[1, 0] == 1 and res.x[2, 0] == 0


def test_weighted_mode_large_lambda_stops_moves():
    p = make_problem(demand=(20, 0, 0), available=(0, 1, 0), retain=0)
    moved = optimize_allocation(p, mode="weighted", travel_penalty_lambda=0.01)
    assert moved.x[1, 0] == 1
    # Moving gains 20 requests but costs lambda * 10 minutes; lambda = 5 makes it not worth it.
    stay = optimize_allocation(p, mode="weighted", travel_penalty_lambda=5.0)
    assert stay.x[1, 0] == 0


def test_coverage_floor_enforced_or_infeasible():
    p = make_problem(demand=(100, 100, 100), available=(3, 0, 0), retain=0, coverage_floor=0.2)
    res = optimize_allocation(p)
    assert res.status == "OPTIMAL"
    assert (compute_outcome(p, res.x).served >= 20).all()
    p_bad = make_problem(demand=(100, 100, 100), available=(3, 0, 0), retain=0, coverage_floor=0.5)
    bad = optimize_allocation(p_bad)
    assert bad.status == "INFEASIBLE"
    assert any("coverage floor" in r for r in bad.diagnostics["infeasibility_reasons"])


def test_status_reports_unknown_or_feasible_with_tiny_time_limit(small_problem):
    res = optimize_allocation(small_problem, time_limit_seconds=1e-6)
    assert res.status in {"OPTIMAL", "FEASIBLE", "UNKNOWN"}
    if res.status == "UNKNOWN":
        assert res.x is None
    else:
        assert validate_allocation(small_problem, res.x) == []


def test_validator_detects_violations(small_problem):
    bad = np.array([[2, 0, 0], [0, 4, 0], [0, 0, 1]])  # origin 1 has only 1 team
    v = validate_allocation(small_problem, bad)
    assert any(x.constraint == "resource conservation" for x in v)
    over = make_problem(available=(6, 0, 0), staging=(5, 5, 5))
    v2 = validate_allocation(over, np.diag(over.available))
    assert any(x.constraint == "staging capacity" for x in v2)
    elig = [[1, 0, 1], [1, 1, 1], [1, 1, 1]]
    blocked = make_problem(eligible=elig)
    x = np.diag(blocked.available)
    x[0, 0], x[0, 1] = 0, 1
    assert any(v.constraint == "route eligibility" for v in validate_allocation(blocked, x))
