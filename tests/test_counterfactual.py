import numpy as np

from resourcemind.allocation.counterfactual import forbid_move, force_move, recommended_moves, value_of_moves
from resourcemind.allocation.models import compute_outcome
from resourcemind.allocation.optimizer import optimize_allocation
from resourcemind.allocation.validation import validate_allocation
from tests.conftest import make_problem


def _plan(p):
    return optimize_allocation(p).x


def test_force_move_is_never_better_than_the_optimum(config):
    p = make_problem(demand=(100, 20, 60), available=(1, 4, 1), retain=1)
    x = _plan(p)
    for j in range(p.n):
        for i in range(p.n):
            if i == j:
                continue
            cf = force_move(p, x, j, i, 1, config)
            if cf.feasible:
                assert cf.x[j, i] >= 1
                assert validate_allocation(p, cf.x) == []
                assert cf.delta_weighted_unmet >= -1e-9  # forcing can't beat the proven optimum


def test_force_move_reports_exact_deltas(config):
    p = make_problem(demand=(20, 0, 0), available=(0, 1, 1), retain=0)  # zone 1 needs one team
    x = _plan(p)  # cheapest: zone 2 (10 min) sends it
    cf = force_move(p, x, 2, 0, 1, config)  # force the 20-min team from zone 3 instead
    assert cf.feasible
    assert cf.delta_weighted_unmet == 0.0
    a, b = compute_outcome(p, x), compute_outcome(p, cf.x)
    assert cf.delta_travel == b.travel_minutes - a.travel_minutes == 10.0
    assert "add 10 team-minutes" in cf.answer


def test_force_move_refuses_ineligible_or_impossible_moves(config):
    p = make_problem(available=(2, 2, 2), max_travel=15, retain=1)  # 1->3 takes 20 min
    x = _plan(p)
    too_far = force_move(p, x, 0, 2, 1, config)
    assert not too_far.feasible and "repositioning limit" in too_far.answer
    too_many = force_move(p, x, 1, 0, 2, config)  # zone 2 has 2 teams and must keep 1
    assert not too_many.feasible and "at most 1 can leave" in too_many.answer
    same = force_move(p, x, 1, 1, 1, config)
    assert not same.feasible


def test_forbid_move_shows_cost_of_an_essential_move(config):
    # Only zone 2 can serve zone 1's demand; forbidding that move must cost service.
    elig = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    p = make_problem(demand=(40, 0, 0), available=(0, 2, 0), eligible=elig, retain=0)
    x = _plan(p)
    assert x[1, 0] == 2
    cf = forbid_move(p, x, 1, 0, config)
    assert cf.feasible
    assert cf.x[1, 0] == 0
    assert cf.delta_served == -40 and cf.delta_weighted_unmet == 40.0


def test_value_of_moves_covers_every_recommended_move(config, dataset, forecast_eval):
    from resourcemind.pipeline import PipelineSettings, run_pipeline

    res = run_pipeline(config, PipelineSettings("surge"), dataset=dataset, forecast_eval=forecast_eval)
    x = res.runs["optimized"].result.x
    table = value_of_moves(res.problem, x, config)
    assert len(table) == len(recommended_moves(res.problem, x))
    # The recommended plan is optimal, so forbidding any move can never improve it.
    assert (table["weighted_unmet_if_forbidden"].dropna() >= -1e-9).all()


def test_move_bounds_on_ineligible_route_is_infeasible():
    p = make_problem(available=(1, 1, 1), max_travel=15)
    assert not p.eligible[0, 2]
    res = optimize_allocation(p, move_bounds={(0, 2): (1, 1)})
    assert res.status == "INFEASIBLE" and res.x is None
    ok = optimize_allocation(p, move_bounds={(0, 1): (0, 0)})
    assert ok.x is not None and ok.x[0, 1] == 0
    np.testing.assert_array_equal(ok.x.sum(axis=1), p.available)
