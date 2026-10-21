import numpy as np
import pytest

from resourcemind.allocation.baselines import greedy_daily
from resourcemind.allocation.multiperiod import compute_daily_outcome, optimize_daily
from resourcemind.allocation.optimizer import optimize_allocation
from resourcemind.allocation.validation import validate_allocation, validate_daily_plan
from resourcemind.simulation.scaled import scaled_problems


def test_scaled_problems_have_consistent_structure():
    single, daily = scaled_problems(30, seed=5)
    assert single.n == daily.n == 30
    np.testing.assert_array_equal(single.demand, daily.demand.sum(axis=1))
    np.testing.assert_array_equal(single.available, daily.available)
    assert (single.available >= 1).all()
    assert (single.staging_capacity >= single.available).all()
    assert np.diag(single.eligible).all() and (np.diag(single.travel) == 0).all()
    assert (single.travel[single.eligible] <= single.max_travel_minutes).all()
    a, b = scaled_problems(30, seed=5)
    np.testing.assert_array_equal(a.demand, single.demand)
    with pytest.raises(ValueError):
        scaled_problems(1)


def test_single_period_solves_optimally_at_50_zones():
    single, _ = scaled_problems(50, seed=1)
    res = optimize_allocation(single, time_limit_seconds=30)
    assert res.status == "OPTIMAL"
    assert validate_allocation(single, res.x) == []
    assert res.diagnostics["weighted_unmet_gap"] == pytest.approx(0.0, abs=1e-6)


def test_warm_started_daily_optimizer_is_never_worse_than_greedy():
    _, daily = scaled_problems(25, seed=3, horizon=3)
    g = greedy_daily(daily)
    res = optimize_daily(daily, time_limit_seconds=5, hint=g.plan)
    assert res.plan is not None and validate_daily_plan(daily, res.plan) == []
    assert "MODEL_INVALID" not in res.stage_statuses.values()
    g_wu = compute_daily_outcome(daily, g.plan).weighted_unmet
    assert compute_daily_outcome(daily, res.plan).weighted_unmet <= g_wu + 1e-9
