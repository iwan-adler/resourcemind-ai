import numpy as np
import pandas as pd
import pytest

from resourcemind.pipeline import PipelineSettings, run_pipeline
from resourcemind.simulation.scenarios import (
    PRESETS,
    ScenarioOverrides,
    apply_scenario,
    merge_overrides,
    scale_teams,
)
from resourcemind.simulation.sensitivity import sensitivity_sweep


def test_presets_exist():
    assert {"normal", "surge", "shortage", "disruption"} <= set(PRESETS)


def test_overrides_do_not_mutate_dataset(config, dataset):
    before_hist = dataset.history.copy(deep=True)
    before_zones = dataset.zones.copy(deep=True)
    before_travel = dataset.travel.copy(deep=True)
    ov = ScenarioOverrides(
        resource_factor=0.5,
        travel_multiplier=2.0,
        priority_weights={"Z01": 3.0},
        blocked_routes=(("Z01", "Z02"),),
    )
    for key in PRESETS:
        apply_scenario(dataset, config, key, ov)
    pd.testing.assert_frame_equal(dataset.history, before_hist)
    pd.testing.assert_frame_equal(dataset.zones, before_zones)
    pd.testing.assert_frame_equal(dataset.travel, before_travel)


def test_reset_restores_defaults(config, dataset):
    default = apply_scenario(dataset, config, "normal", ScenarioOverrides())
    changed = apply_scenario(dataset, config, "normal", ScenarioOverrides(resource_factor=0.5))
    reset = apply_scenario(dataset, config, "normal", ScenarioOverrides())
    assert changed.zones["teams_available"].sum() < default.zones["teams_available"].sum()
    pd.testing.assert_frame_equal(reset.zones, default.zones)
    assert ScenarioOverrides().is_default()
    assert reset.changes == []


def test_surge_changes_history_and_forecast(config, dataset, forecast_eval):
    normal = run_pipeline(config, PipelineSettings("normal"), dataset=dataset, forecast_eval=forecast_eval)
    surge = run_pipeline(config, PipelineSettings("surge"), dataset=dataset, forecast_eval=forecast_eval)
    zones = list(PRESETS["surge"].overrides.history_surge_zones)
    n_hist = normal.inputs.demand_wide()[zones].iloc[-7:].sum().sum()
    s_hist = surge.inputs.demand_wide()[zones].iloc[-7:].sum().sum()
    assert s_hist > 1.5 * n_hist
    n_fc = normal.forecast.totals.set_index("zone_id").loc[zones, "forecast_total"]
    s_fc = surge.forecast.totals.set_index("zone_id").loc[zones, "forecast_total"]
    assert (s_fc > n_fc).all()  # forecast model really sees the surge
    assert surge.problem.demand.sum() > normal.problem.demand.sum()


def test_shortage_propagates(config, dataset, forecast_eval):
    normal = run_pipeline(config, PipelineSettings("normal"), dataset=dataset, forecast_eval=forecast_eval)
    short = run_pipeline(config, PipelineSettings("shortage"), dataset=dataset, forecast_eval=forecast_eval)
    assert short.problem.available.sum() == round(normal.problem.available.sum() * 0.6)
    assert short.runs["optimized"].outcome.unmet.sum() > normal.runs["optimized"].outcome.unmet.sum()


def test_disruption_blocks_inaccessible_zone(config, dataset, forecast_eval):
    res = run_pipeline(config, PipelineSettings("disruption"), dataset=dataset, forecast_eval=forecast_eval)
    idx = res.problem.zone_ids.index("Z02")
    x = res.runs["optimized"].result.x
    assert x[idx, :].sum() == x[idx, idx]  # no team leaves
    assert x[:, idx].sum() == x[idx, idx]  # no team arrives
    assert not res.runs["optimized"].violations


def test_user_demand_multiplier_changes_planning_demand(config, dataset, forecast_eval):
    base = run_pipeline(config, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    up = run_pipeline(
        config,
        PipelineSettings(overrides=ScenarioOverrides(demand_multipliers={"Z04": 3.0})),
        dataset=dataset,
        forecast_eval=forecast_eval,
    )
    i = base.problem.zone_ids.index("Z04")
    assert up.problem.demand[i] == round(base.demand.loc[i, "forecast_total"] * 3.0)
    others = [k for k in range(base.problem.n) if k != i]
    np.testing.assert_array_equal(up.problem.demand[others], base.problem.demand[others])


def test_merge_overrides_layers_user_on_preset():
    merged = merge_overrides(
        PRESETS["disruption"].overrides, ScenarioOverrides(inaccessible_zones=("Z05",), resource_factor=0.8)
    )
    assert set(merged.inaccessible_zones) == {"Z02", "Z05"}
    assert merged.cross_river_multiplier == 3.0
    assert merged.resource_factor == 0.8


def test_invalid_overrides_rejected(config, dataset):
    inputs = apply_scenario(dataset, config, "normal", ScenarioOverrides(inaccessible_zones=("Z99",)))
    assert not inputs.validation.ok
    with pytest.raises(KeyError):
        apply_scenario(dataset, config, "nonexistent")


def test_scale_teams_hits_exact_total():
    t = np.array([4, 2, 4, 1, 4])
    for f in [0, 0.33, 0.6, 1.0, 1.7]:
        assert scale_teams(t, f).sum() == round(t.sum() * f)


def test_overrides_roundtrip_dict():
    ov = ScenarioOverrides(
        blocked_routes=(("Z01", "Z02"),), inaccessible_zones=("Z03",), demand_multipliers={"Z01": 1.5}
    )
    assert ScenarioOverrides.from_dict(ov.to_dict()) == ov


def test_sensitivity_monotone_in_resources(config, dataset, forecast_eval):
    res = run_pipeline(config, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    sweep = sensitivity_sweep(res.problem, config, "resources", [0.5, 0.75, 1.0, 1.25])
    opt = sweep[sweep["strategy"] == "optimized"].sort_values("factor")
    assert opt["weighted_unmet"].is_monotonic_decreasing
    assert set(sweep["strategy"]) == {"current", "greedy", "optimized"}
