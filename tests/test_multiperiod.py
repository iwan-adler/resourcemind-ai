"""Day-by-day (multi-period) allocation: optimizer, baselines, validator and pipeline mode."""

import itertools

import numpy as np

from resourcemind.allocation.baselines import current_daily, greedy_daily
from resourcemind.allocation.multiperiod import (
    DailyPlan,
    MultiPeriodProblem,
    compute_daily_outcome,
    daily_movements_table,
    optimize_daily,
)
from resourcemind.allocation.validation import validate_daily_plan
from resourcemind.pipeline import PipelineSettings, run_pipeline


def make_daily_problem(
    demand=((40, 10), (0, 30)),
    available=(2, 1),
    staging=None,
    service_cap=None,
    weights=None,
    travel=((0, 60), (60, 0)),
    eligible=None,
    retain=0,
    rate=10,
    max_travel=120,
    coverage_floor=0.0,
) -> MultiPeriodProblem:
    """Small hand-checkable problem. demand[i][d]; rate = requests per team per day."""
    demand = np.array(demand, dtype=int)
    n = demand.shape[0]
    travel = np.array(travel, dtype=int)[:n, :n]
    eligible = np.array(travel <= max_travel if eligible is None else eligible, dtype=bool)
    np.fill_diagonal(eligible, True)
    return MultiPeriodProblem(
        zone_ids=tuple(f"Z{i + 1:02d}" for i in range(n)),
        zone_names=tuple(f"Zone {i + 1}" for i in range(n)),
        priority_tiers=tuple("standard" for _ in range(n)),
        demand=demand,
        available=np.array(available, dtype=int),
        staging_capacity=np.array(staging if staging is not None else [5] * n, dtype=int),
        service_cap=np.array(service_cap if service_cap is not None else [10_000] * n, dtype=int),
        weights=np.array(weights if weights is not None else [1.0] * n, dtype=float),
        travel=travel,
        eligible=eligible,
        retain=np.full(n, retain, dtype=int),
        service_per_team=rate,
        max_travel_minutes=max_travel,
        coverage_floor=coverage_floor,
    )


def _brute_force_best(problem):
    """Exhaustive search over every move schedule of a 2-zone problem (ground truth)."""
    total = int(problem.available.sum())
    H = problem.horizon
    best = None
    for values in itertools.product(range(total + 1), repeat=2 * H):
        moves = np.zeros((H, 2, 2), dtype=int)
        for d in range(H):
            moves[d, 0, 1], moves[d, 1, 0] = values[2 * d], values[2 * d + 1]
        plan = DailyPlan.from_moves(problem.available, moves)
        if validate_daily_plan(problem, plan):
            continue
        o = compute_daily_outcome(problem, plan)
        key = (round(o.weighted_unmet, 6), o.travel_minutes)
        if best is None or key < best:
            best = key
    return best


def _key(problem, plan):
    o = compute_daily_outcome(problem, plan)
    return (round(o.weighted_unmet, 6), o.travel_minutes)


def test_matches_brute_force_on_2_zone_2_day_cases():
    cases = [
        make_daily_problem(),
        make_daily_problem(demand=((5, 40), (30, 0)), available=(1, 2), weights=(2.0, 1.0)),
        make_daily_problem(
            demand=((2, 10), (10, 0)), available=(1, 0), weights=(2.0, 1.0), travel=((0, 240), (240, 0)), max_travel=300
        ),
        make_daily_problem(demand=((30, 30), (30, 30)), available=(3, 0), staging=(3, 2), retain=1),
        make_daily_problem(demand=((50, 50), (0, 0)), available=(1, 2), service_cap=(25, 100)),
        make_daily_problem(demand=((20, 0), (0, 20)), available=(0, 2), eligible=((1, 0), (0, 1))),
    ]
    for p in cases:
        res = optimize_daily(p)
        assert res.status == "OPTIMAL"
        assert validate_daily_plan(p, res.plan) == []
        assert _key(p, res.plan) == _brute_force_best(p)


def test_travel_time_consumes_service_on_the_moving_day():
    # One team, half a shift of travel: it serves floor(10 * (1 - 240/480)) = 5 on arrival, 10 afterwards.
    p = make_daily_problem(demand=((0, 0), (10, 10)), available=(1, 0), travel=((0, 240), (240, 0)), max_travel=300)
    res = optimize_daily(p)
    o = compute_daily_outcome(p, res.plan)
    assert res.plan.moves[0, 0, 1] == 1
    np.testing.assert_array_equal(o.served[1], [5, 10])
    mv = daily_movements_table(p, res.plan)
    assert list(mv["day"]) == [1] and mv["service_lost_requests"].iloc[0] == 5.0


def test_conservation_staging_and_eligibility_hold():
    p = make_daily_problem(
        demand=((80, 80, 0), (0, 0, 80), (80, 0, 80)),
        available=(1, 4, 1),
        staging=(3, 5, 3),
        travel=((0, 20, 30), (20, 0, 25), (30, 25, 0)),
        eligible=((1, 1, 0), (1, 1, 1), (0, 1, 1)),
        retain=1,
    )
    res = optimize_daily(p)
    assert res.status == "OPTIMAL"
    assert validate_daily_plan(p, res.plan) == []
    plan = res.plan
    np.testing.assert_array_equal(plan.stationed.sum(axis=0), np.full(p.horizon, p.available.sum()))
    assert (plan.stationed <= p.staging_capacity[:, None]).all()
    assert plan.moves[:, 0, 2].sum() == 0 and plan.moves[:, 2, 0].sum() == 0
    np.testing.assert_array_equal(DailyPlan.from_moves(p.available, plan.moves).stationed, plan.stationed)


def test_validator_detects_violations():
    p = make_daily_problem(staging=(5, 2), eligible=((1, 0), (1, 1)), retain=1)
    ok = DailyPlan.from_moves(p.available, np.zeros((2, 2, 2), dtype=int))
    assert validate_daily_plan(p, ok) == []

    bad_conservation = DailyPlan(stationed=np.array([[2, 2], [2, 2]]), moves=np.zeros((2, 2, 2), dtype=int))
    assert any(v.constraint == "resource conservation" for v in validate_daily_plan(p, bad_conservation))

    moves = np.zeros((2, 2, 2), dtype=int)
    moves[0, 0, 1] = 1  # route 1 -> 2 is not eligible
    assert any(
        v.constraint == "route eligibility" for v in validate_daily_plan(p, DailyPlan.from_moves(p.available, moves))
    )

    moves = np.zeros((2, 2, 2), dtype=int)
    moves[0, 1, 0] = 1  # zone 2 must keep its single team
    assert any(
        v.constraint == "minimum retention" for v in validate_daily_plan(p, DailyPlan.from_moves(p.available, moves))
    )

    over = make_daily_problem(available=(1, 3), staging=(5, 2))
    stay = DailyPlan.from_moves(over.available, np.zeros((2, 2, 2), dtype=int))
    assert any(v.constraint == "staging capacity" for v in validate_daily_plan(over, stay))


def test_infeasible_reported_honestly():
    p = make_daily_problem(available=(3, 3), staging=(2, 2))
    res = optimize_daily(p)
    assert res.status == "INFEASIBLE"
    assert res.plan is None
    assert any("staging capacity" in r for r in res.diagnostics["infeasibility_reasons"])


def test_zero_demand_and_zero_resources():
    p0 = make_daily_problem(demand=((0, 0), (0, 0)))
    r0 = optimize_daily(p0)
    assert r0.status == "OPTIMAL"
    assert compute_daily_outcome(p0, r0.plan).teams_moved == 0

    pz = make_daily_problem(available=(0, 0))
    rz = optimize_daily(pz)
    assert rz.status == "OPTIMAL"
    o = compute_daily_outcome(pz, rz.plan)
    assert o.served.sum() == 0 and o.unmet.sum() == pz.demand.sum()


def test_optimizer_is_deterministic():
    p = make_daily_problem(
        demand=((30, 5, 40), (10, 40, 0), (25, 25, 25)),
        available=(2, 2, 2),
        travel=((0, 20, 30), (20, 0, 25), (30, 25, 0)),
    )
    a, b = optimize_daily(p, seed=3), optimize_daily(p, seed=3)
    np.testing.assert_array_equal(a.plan.moves, b.plan.moves)
    np.testing.assert_array_equal(a.plan.stationed, b.plan.stationed)


def test_greedy_valid_and_optimizer_never_worse_on_random_instances():
    rng = np.random.default_rng(11)
    travel = ((0, 20, 40), (20, 0, 30), (40, 30, 0))
    for _ in range(20):
        p = make_daily_problem(
            demand=rng.integers(0, 50, (3, 3)),
            available=tuple(rng.integers(0, 4, 3)),
            staging=tuple(rng.integers(4, 8, 3)),
            weights=tuple(np.round(rng.uniform(1, 2, 3), 2)),
            travel=travel,
            max_travel=35,
            retain=int(rng.integers(0, 2)),
        )
        greedy = greedy_daily(p)
        assert validate_daily_plan(p, greedy.plan) == [], greedy.messages
        opt = optimize_daily(p)
        assert opt.status == "OPTIMAL"
        assert validate_daily_plan(p, opt.plan) == []
        g = compute_daily_outcome(p, greedy.plan).weighted_unmet
        c = compute_daily_outcome(p, current_daily(p).plan).weighted_unmet
        o = compute_daily_outcome(p, opt.plan).weighted_unmet
        assert o <= g + 1e-9 and o <= c + 1e-9


def test_optimizer_anticipates_where_myopic_greedy_does_not():
    # Day 1 tempts greedy to send zone 1's team to zone 2; day 2 needs it back, and each
    # move costs half a shift. Staying put is strictly better.
    p = make_daily_problem(
        demand=((2, 10), (10, 0)), available=(1, 0), weights=(2.0, 1.0), travel=((0, 240), (240, 0)), max_travel=300
    )
    g = compute_daily_outcome(p, greedy_daily(p).plan)
    o = compute_daily_outcome(p, optimize_daily(p).plan)
    assert g.teams_moved == 2
    assert o.teams_moved == 0
    assert o.weighted_unmet < g.weighted_unmet


def test_pipeline_daily_mode(config, dataset, forecast_eval):
    single = run_pipeline(config, PipelineSettings("surge"), dataset=dataset, forecast_eval=forecast_eval)
    assert single.daily is None and single.planning_mode == "single"
    res = run_pipeline(
        config.with_updates(planning_mode="daily"),
        PipelineSettings("surge"),
        dataset=dataset,
        forecast_eval=forecast_eval,
    )
    d = res.daily
    assert res.planning_mode == "daily"
    assert d.problem.demand.shape == (res.problem.n, config.horizon_days)
    assert list(d.comparison.columns) == list(single.comparison.columns)
    opt = d.runs["optimized"]
    assert opt.result.status == "OPTIMAL" and opt.violations == []
    for key in ["current", "greedy"]:
        assert d.runs[key].violations == []
        assert opt.outcome.weighted_unmet <= d.runs[key].outcome.weighted_unmet + 1e-9
    # single-period results are unchanged by the daily mode
    np.testing.assert_array_equal(res.runs["optimized"].result.x, single.runs["optimized"].result.x)
