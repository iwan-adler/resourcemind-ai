import numpy as np
import pandas as pd
import pytest

from resourcemind.config import ForecastConfig
from resourcemind.forecasting.evaluation import evaluate_forecasters, forecast_future, split_dates
from resourcemind.forecasting.features import InsufficientHistoryError, build_supervised
from resourcemind.forecasting.models import MODEL_REGISTRY, make_model


def _wide(values: np.ndarray, start="2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=values.shape[0], freq="D")
    return pd.DataFrame(values, index=dates, columns=[f"Z{i + 1:02d}" for i in range(values.shape[1])])


def test_features_use_only_past_data():
    # Demand = day index, so any feature value > origin index would reveal future data.
    n = 60
    wide = _wide(np.tile(np.arange(n, dtype=float)[:, None], (1, 2)))
    rows = build_supervised(wide, horizon=7)
    origin_idx = (rows["origin_date"] - wide.index[0]).dt.days
    for col in ["last_value", "mean_7", "mean_28", "lag_dow_1", "lag_dow_2", "lag_dow_3"]:
        assert (rows[col] <= origin_idx + 1e-9).all(), col
    assert ((rows["target_date"] - rows["origin_date"]).dt.days == rows["horizon"]).all()
    assert (rows["y"] == origin_idx + rows["horizon"]).all()


def test_split_is_chronological_and_disjoint(dataset, config):
    rows = build_supervised(dataset.demand_wide(), 7)
    d = split_dates(pd.Timestamp(dataset.demand_wide().index.max()), config.forecast)
    train = rows[rows["target_date"] < d["val_start"]]
    test = rows[rows["target_date"] >= d["test_start"]]
    assert train["target_date"].max() < d["val_start"] <= d["test_start"]
    assert train["target_date"].max() < test["target_date"].min()


def test_evaluation_outputs(forecast_eval):
    assert set(forecast_eval.test_metrics["model"]) == set(MODEL_REGISTRY)
    assert forecast_eval.selected_model in MODEL_REGISTRY
    best_val = forecast_eval.validation_metrics.sort_values("mae").iloc[0]["model"]
    assert forecast_eval.selected_model == best_val  # selection uses validation, never test
    assert (forecast_eval.test_metrics[["mae", "rmse"]] >= 0).all().all()
    assert (forecast_eval.test_metrics["rmse"] >= forecast_eval.test_metrics["mae"] - 1e-9).all()
    for cov in forecast_eval.test_interval_coverage.values():
        assert 0.0 <= cov <= 1.0


def test_future_forecast_has_every_zone_and_horizon(dataset, forecast_eval):
    wide = dataset.demand_wide()
    for model in MODEL_REGISTRY:
        fc = forecast_future(wide, model, 5, forecast_eval.interval_quantiles[model])
        assert len(fc.daily) == wide.shape[1] * 5
        assert set(fc.daily["zone_id"]) == set(wide.columns)
        assert sorted(fc.daily["horizon"].unique()) == [1, 2, 3, 4, 5]
        assert (fc.daily["target_date"] > wide.index.max()).all()
        assert (fc.daily["predicted"] >= 0).all()
        assert (fc.daily["lower"] <= fc.daily["predicted"] + 1e-9).all()
        assert (fc.daily["upper"] >= fc.daily["predicted"] - 1e-9).all()
        assert len(fc.totals) == wide.shape[1]


def test_future_forecast_ignores_values_after_origin(dataset, forecast_eval):
    """Truncating history at T must give the same forecast regardless of later data."""
    wide = dataset.demand_wide()
    cut = wide.iloc[:200]
    q = forecast_eval.interval_quantiles["gradient_boosting"]
    a = forecast_future(cut, "gradient_boosting", 7, q)
    tampered = wide.copy()
    tampered.iloc[200:] = 10_000  # future values that must not influence the forecast at day 200
    b = forecast_future(tampered.iloc[:200], "gradient_boosting", 7, q)
    np.testing.assert_allclose(a.daily["predicted"], b.daily["predicted"])


def test_constant_series_forecast_is_constant():
    wide = _wide(np.full((40, 3), 7.0))
    for name in MODEL_REGISTRY:
        fc = forecast_future(wide, name, 3, {"daily": (0.0, 0.0), "total": (0.0, 0.0)})
        np.testing.assert_allclose(fc.daily["predicted"], 7.0)


def test_all_zero_series_is_safe():
    wide = _wide(np.zeros((40, 2)))
    fc = forecast_future(wide, "gradient_boosting", 2, {"daily": (0.0, 0.0)})
    assert (fc.daily["predicted"] == 0).all()


def test_too_short_history_raises_clear_error():
    with pytest.raises(InsufficientHistoryError):
        build_supervised(_wide(np.ones((20, 2))), horizon=3)


def test_small_dataset_evaluation_runs():
    rng = np.random.default_rng(0)
    wide = _wide(rng.poisson(10, size=(90, 2)).astype(float))
    ev = evaluate_forecasters(
        wide,
        3,
        ForecastConfig(validation_days=14, test_days=14),
        models=["seasonal_naive", "gradient_boosting"],
    )
    assert len(ev.test_metrics) == 2


def test_invalid_horizon_rejected(dataset):
    with pytest.raises(ValueError):
        build_supervised(dataset.demand_wide(), horizon=8)


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        make_model("prophet")
