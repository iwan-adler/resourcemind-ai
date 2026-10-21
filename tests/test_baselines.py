import numpy as np

from resourcemind.allocation.baselines import current_allocation, greedy_allocation
from resourcemind.allocation.models import compute_outcome
from resourcemind.allocation.optimizer import optimize_allocation
from resourcemind.allocation.validation import validate_allocation
from resourcemind.pipeline import PipelineSettings, run_pipeline
from tests.conftest import make_problem


def test_current_allocation_keeps_everyone(small_problem):
    res = current_allocation(small_problem)
    np.testing.assert_array_equal(res.x, np.diag(small_problem.available))
    assert compute_outcome(small_problem, res.x).teams_moved == 0


def test_greedy_respects_hard_constraints():
    cases = [
        make_problem(),
        make_problem(demand=(200, 0, 0), available=(0, 5, 5), staging=(3, 5, 7), retain=1),
        make_problem(demand=(100, 0, 0), available=(0, 4, 4), eligible=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        make_problem(demand=(100, 0, 0), available=(0, 0, 4), max_travel=15),
    ]
    for p in cases:
        res = greedy_allocation(p)
        assert validate_allocation(p, res.x) == [], res.messages


def test_greedy_is_deterministic(small_problem):
    a, b = greedy_allocation(small_problem), greedy_allocation(small_problem)
    np.testing.assert_array_equal(a.x, b.x)


def test_greedy_never_worse_than_current_and_optimizer_never_worse_than_greedy():
    rng = np.random.default_rng(3)
    for _ in range(25):
        p = make_problem(
            demand=tuple(rng.integers(0, 120, 3)),
            available=tuple(rng.integers(0, 4, 3)),
            staging=tuple(rng.integers(4, 9, 3)),
            weights=tuple(np.round(rng.uniform(1, 2, 3), 2)),
            retain=int(rng.integers(0, 2)),
        )
        cur = compute_outcome(p, current_allocation(p).x).weighted_unmet
        gr = compute_outcome(p, greedy_allocation(p).x).weighted_unmet
        opt = optimize_allocation(p)
        assert opt.status == "OPTIMAL"
        assert gr <= cur + 1e-9
        assert compute_outcome(p, opt.x).weighted_unmet <= gr + 1e-9


def test_all_strategies_share_identical_inputs(config, dataset, forecast_eval):
    result = run_pipeline(config, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    p = result.problem
    for run in result.runs.values():
        # every strategy is scored by the same outcome function on the same problem
        o = compute_outcome(p, run.result.x)
        assert o.weighted_unmet == run.outcome.weighted_unmet
        np.testing.assert_array_equal(run.result.x.sum(axis=1), p.available)


def test_greedy_reports_unrepairable_staging_violation():
    p = make_problem(available=(6, 0, 0), staging=(5, 5, 5), eligible=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    res = greedy_allocation(p)
    assert any("above staging capacity" in m for m in res.messages)
    assert any(v.constraint == "staging capacity" for v in validate_allocation(p, res.x))
