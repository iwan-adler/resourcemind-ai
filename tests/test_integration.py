"""End-to-end: data -> forecast -> gap analysis -> allocation -> validation -> evaluation."""

import numpy as np

from resourcemind.pipeline import PipelineSettings, run_pipeline
from resourcemind.simulation.scenarios import PRESETS, ScenarioOverrides


def test_full_workflow_every_preset(config, dataset, forecast_eval):
    for key in PRESETS:
        res = run_pipeline(config, PipelineSettings(key), dataset=dataset, forecast_eval=forecast_eval)
        p = res.problem
        # gaps are computed from the same scenario inputs the allocator sees
        np.testing.assert_array_equal(res.gaps["planning_demand"], p.demand)
        np.testing.assert_array_equal(res.gaps["gap"], p.demand - p.available * p.service_per_team)
        opt = res.runs["optimized"]
        assert opt.result.status == "OPTIMAL", key
        assert opt.violations == []
        assert res.rationale is not None and len(res.rationale) == p.n
        for strategy in ["current", "greedy"]:
            assert res.runs[strategy].violations == []
            assert opt.outcome.weighted_unmet <= res.runs[strategy].outcome.weighted_unmet + 1e-9


def test_pipeline_is_reproducible(config, dataset, forecast_eval):
    a = run_pipeline(config, PipelineSettings("surge"), dataset=dataset, forecast_eval=forecast_eval)
    b = run_pipeline(config, PipelineSettings("surge"), dataset=dataset, forecast_eval=forecast_eval)
    np.testing.assert_array_equal(a.runs["optimized"].result.x, b.runs["optimized"].result.x)
    np.testing.assert_allclose(a.forecast.totals["forecast_total"], b.forecast.totals["forecast_total"])


def test_infeasible_scenario_flows_through_pipeline(config, dataset, forecast_eval):
    res = run_pipeline(
        config,
        PipelineSettings(overrides=ScenarioOverrides(staging_capacity_factor=0.3)),
        dataset=dataset,
        forecast_eval=forecast_eval,
    )
    opt = res.runs["optimized"]
    assert opt.result.status == "INFEASIBLE"
    assert opt.outcome is None
    assert res.rationale is None
    assert any("staging" in line for line in res.summary_lines)
    # current allocation now violates staging capacity and is reported as such
    assert len(res.runs["current"].violations) > 0
    assert res.comparison.set_index("strategy").loc["current", "hard_violations"] > 0


def test_planning_basis_upper_increases_demand(config, dataset, forecast_eval):
    point = run_pipeline(config, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    upper = run_pipeline(
        config.with_updates(planning_basis="upper"),
        PipelineSettings(),
        dataset=dataset,
        forecast_eval=forecast_eval,
    )
    assert upper.problem.demand.sum() > point.problem.demand.sum()
