import itertools
from dataclasses import replace

import numpy as np
import pytest

from resourcemind.allocation.counterfactual import force_move
from resourcemind.allocation.models import compute_outcome
from resourcemind.allocation.optimizer import optimize_allocation, optimize_expected
from resourcemind.allocation.response import best_service, response_pairs
from resourcemind.allocation.validation import validate_allocation
from resourcemind.pipeline import PipelineSettings, run_pipeline
from resourcemind.simulation.robustness import robustness_check
from resourcemind.simulation.scenarios import local_response_minutes
from tests.conftest import make_problem


def _with_pairs(p, pairs):
    return replace(p, response_pairs=np.array(pairs, dtype=bool), response_standard_minutes=15.0)


def test_identity_pairs_reduce_to_the_in_zone_model():
    p = make_problem()
    q = _with_pairs(p, np.eye(3))
    for x in [np.diag(p.available), optimize_allocation(p).x]:
        a, b = compute_outcome(p, x), compute_outcome(q, x)
        np.testing.assert_array_equal(a.served, b.served)
    assert optimize_allocation(q).objective == optimize_allocation(p).objective


def test_neighbour_team_covers_a_zone_with_no_team():
    # Zone 1 has demand but no team; zone 2's team (spare capacity) can reach it in time.
    served, flows = best_service(
        stationed=np.array([0, 1, 0]),
        servable=np.array([15, 5, 0]),
        weights=np.ones(3),
        rate=20,
        pairs=np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
    )
    np.testing.assert_array_equal(served, [15, 5, 0])
    assert flows[1, 0] == 15 and flows.sum(axis=1)[1] <= 20


def test_best_service_respects_capacity_and_prefers_higher_weight():
    served, _ = best_service(
        stationed=np.array([1, 0]),
        servable=np.array([20, 20]),
        weights=np.array([1.0, 2.0]),
        rate=20,
        pairs=np.array([[1, 1], [0, 1]], dtype=bool),
    )
    assert served.sum() == 20 and served[1] == 20  # all capacity goes to the weight-2 zone


def test_standard_below_local_response_covers_nothing():
    pairs = response_pairs(np.array([[0, 30], [30, 0]]), np.array([11.0, 11.0]), 10.0, np.zeros((2, 2), bool))
    assert not pairs.any()
    p = _with_pairs(make_problem(demand=(50, 50, 0), available=(2, 2, 1)), np.zeros((3, 3)))
    o = compute_outcome(p, np.diag(p.available))
    assert o.served.sum() == 0 and o.unmet.sum() == 100
    assert "no team stationed within the response standard" in o.limiting_factor[0]


def test_response_optimizer_matches_brute_force():
    p = _with_pairs(
        make_problem(demand=(60, 30, 40), available=(1, 2, 1), weights=(1.5, 1.0, 1.2), retain=0),
        [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
    )
    res = optimize_allocation(p)
    assert res.status == "OPTIMAL" and validate_allocation(p, res.x) == []
    best = None
    rows = [[r for r in itertools.product(range(a + 1), repeat=3) if sum(r) == a] for a in p.available]
    for combo in itertools.product(*rows):
        x = np.array(combo, dtype=int)
        if validate_allocation(p, x):
            continue
        o = compute_outcome(p, x)
        key = (round(o.weighted_unmet, 6), o.travel_minutes)
        best = key if best is None or key < best else best
    o = compute_outcome(p, res.x)
    assert (round(o.weighted_unmet, 6), o.travel_minutes) == best


def test_pipeline_with_standard_is_valid_and_optimizer_not_worse(config, dataset, forecast_eval):
    for minutes in [15.0, 20.0]:
        cfg = config.with_updates(response_standard_minutes=minutes)
        res = run_pipeline(cfg, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
        assert res.problem.uses_response_standard
        assert res.problem.response_standard_minutes == minutes
        assert np.diag(res.problem.response_pairs).all()  # 11.1-min local response is within both standards
        cmp = res.comparison.set_index("strategy")
        assert (cmp["hard_violations"] == 0).all()
        assert res.runs["optimized"].result.status == "OPTIMAL"
        assert cmp.loc["optimized", "weighted_unmet"] <= cmp.loc["greedy", "weighted_unmet"] + 1e-9
        assert cmp.loc["optimized", "weighted_unmet"] <= cmp.loc["current", "weighted_unmet"] + 1e-9


def test_standard_off_leaves_results_unchanged(config, dataset, forecast_eval):
    res = run_pipeline(config, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    assert not res.problem.uses_response_standard
    assert res.comparison.set_index("strategy").loc["optimized", "weighted_unmet"] == pytest.approx(123.01)


def test_standard_rejected_in_daily_mode(config):
    with pytest.raises(ValueError, match="single-period"):
        config.with_updates(response_standard_minutes=20.0, planning_mode="daily").validate()
    assert 10 < local_response_minutes(10) < 12


def test_robustness_and_what_if_work_with_standard(config, dataset, forecast_eval):
    cfg = config.with_updates(response_standard_minutes=20.0)
    res = run_pipeline(cfg, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    rob = robustness_check(res, n_samples=20, seed=0, include_expected=True)
    assert "optimized_expected" not in set(rob.summary["strategy"])
    assert any("response-time standard" in n for n in rob.notes)
    x = res.runs["optimized"].result.x
    cf = force_move(res.problem, x, 7, 4, 1, cfg)
    assert cf.feasible and cf.delta_weighted_unmet >= -1e-9
    with pytest.raises(NotImplementedError):
        optimize_expected(res.problem, np.ones((2, res.problem.n), dtype=int))
