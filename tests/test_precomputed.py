import pandas as pd

from resourcemind.forecasting import precomputed
from resourcemind.forecasting.evaluation import evaluate_forecasters


def test_shipped_evaluation_matches_a_fresh_computation(config, dataset, forecast_eval):
    stored = precomputed.load(dataset.demand_wide(), config.horizon_days, config.forecast, config.seed)
    assert stored is not None, "data/precomputed is missing or stale: run python scripts/precompute.py"
    assert stored.selected_model == forecast_eval.selected_model
    pd.testing.assert_frame_equal(stored.test_metrics, forecast_eval.test_metrics)
    pd.testing.assert_frame_equal(stored.validation_metrics, forecast_eval.validation_metrics)
    assert stored.interval_quantiles == forecast_eval.interval_quantiles
    for name, frame in forecast_eval.test_total_residuals.items():
        pd.testing.assert_frame_equal(stored.test_total_residuals[name], frame)


def test_every_horizon_is_stored(config, dataset):
    wide = dataset.demand_wide()
    for horizon in range(1, 8):
        assert precomputed.load(wide, horizon, config.forecast, config.seed) is not None


def test_mismatch_falls_back_to_none(tmp_path, config, dataset):
    wide = dataset.demand_wide()
    # different data, different seed, or no file: never returns a stale result
    assert precomputed.load(wide * 2, 7, config.forecast, config.seed) is None
    assert precomputed.load(wide, 7, config.forecast, config.seed + 1) is None
    assert precomputed.load(wide, 7, config.forecast, config.seed, directory=tmp_path) is None


def test_save_and_load_roundtrip(tmp_path, config):
    rng_wide = pd.DataFrame(
        [[5, 7], [6, 8]] * 60, index=pd.date_range("2025-01-01", periods=120), columns=["Z01", "Z02"]
    ).astype(float)
    ev = evaluate_forecasters(rng_wide, 3, config.forecast, models=["seasonal_naive"])
    fp = precomputed.fingerprint(rng_wide, 3, config.forecast, 1)
    precomputed.save({3: (fp, ev)}, 1, directory=tmp_path)
    loaded = precomputed.load(rng_wide, 3, config.forecast, 1, directory=tmp_path)
    pd.testing.assert_frame_equal(loaded.test_metrics, ev.test_metrics)
