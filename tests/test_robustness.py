import numpy as np
import pandas as pd
import pytest

from resourcemind.allocation.models import compute_outcome
from resourcemind.allocation.optimizer import optimize_allocation, optimize_expected
from resourcemind.allocation.validation import validate_allocation
from resourcemind.pipeline import PipelineSettings, run_pipeline
from resourcemind.simulation.robustness import evaluate_plan_on_samples, robustness_check, sample_demand
from tests.conftest import make_problem


def test_evaluate_plan_matches_shared_outcome_for_each_sample():
    p = make_problem()
    x = optimize_allocation(p).x
    samples = np.array([[100, 20, 60], [0, 0, 0], [150, 5, 90]])
    m = evaluate_plan_on_samples(p, x, samples)
    for s, row in enumerate(samples):
        o = compute_outcome(type(p)(**{**p.__dict__, "demand": row}), x)
        assert m["weighted_unmet"][s] == pytest.approx(o.weighted_unmet)
    assert m["coverage"][1] == 1.0  # zero demand counts as fully covered
    assert ((m["coverage"] >= 0) & (m["coverage"] <= 1)).all()


def test_sample_demand_keeps_zone_errors_together_and_is_reproducible():
    residuals = pd.DataFrame([[0.1, 0.1], [-0.2, -0.2]], columns=["A", "B"])
    a = sample_demand(np.array([100.0, 50.0]), np.ones(2), residuals, 50, seed=3)
    b = sample_demand(np.array([100.0, 50.0]), np.ones(2), residuals, 50, seed=3)
    np.testing.assert_array_equal(a, b)
    # each sample uses one window: both zones +10% or both -20%
    assert set(map(tuple, a)) <= {(110, 55), (80, 40)}
    with pytest.raises(ValueError):
        sample_demand(np.ones(2), np.ones(2), pd.DataFrame(), 5, 0)


def test_optimize_expected_is_valid_and_optimal_on_its_own_samples():
    p = make_problem(demand=(100, 20, 60), available=(1, 4, 1), retain=0)
    samples = np.array([[100, 20, 60], [40, 80, 20], [60, 60, 60]])
    res = optimize_expected(p, samples)
    assert res.status == "OPTIMAL"
    assert validate_allocation(p, res.x) == []
    mean_wu = evaluate_plan_on_samples(p, res.x, samples)["weighted_unmet"].mean()
    for other in [optimize_allocation(p).x, np.diag(p.available)]:
        assert mean_wu <= evaluate_plan_on_samples(p, other, samples)["weighted_unmet"].mean() + 1e-9
    with pytest.raises(ValueError):
        optimize_expected(p, np.zeros((0, 3)))


def test_forecast_evaluation_keeps_validation_and_test_residuals(forecast_eval, dataset):
    for name in forecast_eval.test_metrics["model"]:
        val = forecast_eval.validation_total_residuals[name]
        test = forecast_eval.test_total_residuals[name]
        assert list(val.columns) == list(dataset.demand_wide().columns) == list(test.columns)
        assert len(val) > 0 and len(test) > 0
        # validation windows end before the test windows start
        assert val.index.max() < test.index.min()


def test_robustness_check_scores_every_plan_on_held_out_errors(config, dataset, forecast_eval):
    res = run_pipeline(config, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    rob = robustness_check(res, n_samples=60, seed=1)
    assert set(rob.summary["strategy"]) == {"current", "greedy", "optimized_point", "optimized_upper"}
    assert rob.n_test_windows == len(forecast_eval.test_total_residuals[res.forecast.model])
    assert ((rob.summary["mean_coverage"] > 0) & (rob.summary["mean_coverage"] <= 1)).all()
    assert len(rob.coverage_samples) == 60 * len(rob.summary)
    again = robustness_check(res, n_samples=60, seed=1)
    pd.testing.assert_frame_equal(rob.summary, again.summary)
    with_expected = robustness_check(res, n_samples=30, seed=1, include_expected=True)
    assert "optimized_expected" in set(with_expected.summary["strategy"])
