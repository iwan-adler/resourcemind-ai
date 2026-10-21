import json
import math

import numpy as np
import pytest

from resourcemind.metrics.evaluation import (
    coverage_ratio,
    fairness_summary,
    gini,
    regression_metrics,
    tier_table,
    zone_outcome_table,
)
from resourcemind.pipeline import PipelineSettings, run_pipeline, run_record


def test_regression_metrics_known_values():
    m = regression_metrics([10, 20, 30], [12, 18, 30])
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(8 / 3))
    assert m["wape"] == pytest.approx(4 / 60)
    assert m["bias"] == pytest.approx(0.0)
    assert m["n"] == 3


def test_regression_metrics_zero_and_empty():
    zero = regression_metrics([0, 0], [1, 1])
    assert math.isnan(zero["wape"]) and zero["mae"] == 1
    empty = regression_metrics([], [])
    assert empty["n"] == 0 and math.isnan(empty["mae"])
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])


def test_coverage_bounds_and_zero_demand():
    assert coverage_ratio(0, 0) == 1.0
    assert coverage_ratio(5, 10) == 0.5
    assert coverage_ratio(15, 10) == 1.0
    assert coverage_ratio(-1, 10) == 0.0


def test_gini():
    assert gini([]) == 0.0
    assert gini([0, 0]) == 0.0
    assert gini([5, 5, 5]) == pytest.approx(0.0)
    assert gini([0, 0, 10]) > 0.6


def test_zone_table_and_fairness_consistency():
    zt = zone_outcome_table(["A", "B"], ["a", "b"], ["high", "standard"], [2.0, 1.0], [10, 0], [5, 0])
    assert list(zt["unmet"]) == [5, 0]
    assert list(zt["weighted_unmet"]) == [10, 0]
    assert list(zt["coverage"]) == [0.5, 1.0]
    f = fairness_summary(zt)
    assert f["min_zone_coverage"] == 0.5 and f["coverage_range"] == 0.5
    assert f["coverage_tier_high"] == 0.5
    tiers = tier_table(zt)
    assert set(tiers["priority_tier"]) == {"high", "standard"}
    assert fairness_summary(zt.iloc[0:0])["coverage_range"] == 0.0


def test_comparison_table_is_consistent(config, dataset, forecast_eval):
    res = run_pipeline(config, PipelineSettings(), dataset=dataset, forecast_eval=forecast_eval)
    c = res.comparison.set_index("strategy")
    for s in ["current", "greedy", "optimized"]:
        row = c.loc[s]
        assert 0.0 <= row["coverage"] <= 1.0
        assert row["served"] + row["total_unmet"] == pytest.approx(row["total_demand"])
        zt = res.zone_tables[s]
        assert zt["weighted_unmet"].sum() == pytest.approx(row["weighted_unmet"])
    assert c.loc["optimized", "weighted_unmet"] <= c.loc["greedy", "weighted_unmet"] + 1e-9
    assert c.loc["optimized", "hard_violations"] == 0


def test_run_record_is_json_serialisable(config, dataset, forecast_eval):
    res = run_pipeline(config, PipelineSettings("surge"), dataset=dataset, forecast_eval=forecast_eval)
    rec = run_record(res)
    text = json.dumps(rec, default=str)
    assert "synthetic" in text
    for key in ["scenario", "seed", "forecast", "solver", "strategies", "resources"]:
        assert key in rec
    assert rec["solver"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert not np.isnan(rec["strategies"][0]["coverage"])
