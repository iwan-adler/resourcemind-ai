"""Chronological evaluation, empirical intervals and out-of-sample forecasts.

Split by target date (last date = T):
    train      : targets <= T - test_days - validation_days
    validation : the next `validation_days` days of targets
    test       : the final `test_days` days of targets

Procedure:
1. Fit every model on train, predict validation. Validation results select the
   default model and supply the residual quantiles for intervals.
2. Refit on train + validation, predict test. Test results are reported and
   never used for any choice.
3. For the live forecast, refit on all labelled rows and predict past T.

Features only use data up to each row's origin date, so no future values leak in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from resourcemind.config import ForecastConfig
from resourcemind.forecasting.features import build_supervised
from resourcemind.forecasting.models import MODEL_REGISTRY, make_model
from resourcemind.metrics.evaluation import regression_metrics


@dataclass
class ForecastEvaluation:
    horizon: int
    split: dict[str, str]
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    test_per_zone: pd.DataFrame
    test_per_horizon: pd.DataFrame
    test_predictions: pd.DataFrame
    interval_quantiles: dict[str, dict[str, tuple[float, float]]]
    test_interval_coverage: dict[str, float]
    selected_model: str
    runtime_seconds: dict[str, float] = field(default_factory=dict)
    # Per model: relative errors of horizon-total forecasts on the validation window,
    # one row per forecast origin and one column per zone (used by the robustness check).
    validation_total_residuals: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Same for the held-out test window (model refitted on train + validation); only used
    # to score plans out of sample.
    test_total_residuals: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class ZoneForecast:
    model: str
    origin_date: pd.Timestamp
    daily: pd.DataFrame  # zone_id, target_date, horizon, predicted, lower, upper
    totals: pd.DataFrame  # zone_id, forecast_total, lower_total, upper_total
    interval_note: str


def _relative_residuals(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (y - p) / np.maximum(p, 1.0)


def _interval_quantiles(pred_rows: pd.DataFrame, horizon: int, lo: float, hi: float) -> dict:
    rel = _relative_residuals(pred_rows["y"].to_numpy(), pred_rows["pred"].to_numpy())
    daily = (float(np.quantile(rel, lo)), float(np.quantile(rel, hi))) if rel.size else (0.0, 0.0)
    sums = pred_rows.groupby(["zone_id", "origin_date"]).agg(
        y=("y", "sum"), pred=("pred", "sum"), k=("horizon", "count")
    )
    sums = sums[sums["k"] == horizon]
    rel_t = _relative_residuals(sums["y"].to_numpy(), sums["pred"].to_numpy())
    total = (float(np.quantile(rel_t, lo)), float(np.quantile(rel_t, hi))) if rel_t.size else daily
    return {"daily": daily, "total": total}


def _total_residual_matrix(pred_rows: pd.DataFrame, horizon: int, zone_ids: list[str]) -> pd.DataFrame:
    """Relative horizon-total errors as an (origin x zone) matrix; only origins complete for every zone."""
    sums = pred_rows.groupby(["origin_date", "zone_id"]).agg(
        y=("y", "sum"), pred=("pred", "sum"), k=("horizon", "count")
    )
    sums = sums[sums["k"] == horizon]
    sums["rel"] = _relative_residuals(sums["y"].to_numpy(), sums["pred"].to_numpy())
    matrix = sums["rel"].unstack("zone_id").reindex(columns=zone_ids)
    return matrix.dropna()


def split_dates(last_date: pd.Timestamp, config: ForecastConfig) -> dict[str, pd.Timestamp]:
    test_start = last_date - pd.Timedelta(days=config.test_days - 1)
    val_start = test_start - pd.Timedelta(days=config.validation_days)
    return {"val_start": val_start, "test_start": test_start, "last_date": last_date}


def evaluate_forecasters(
    demand_wide: pd.DataFrame,
    horizon: int,
    config: ForecastConfig,
    models: list[str] | None = None,
    random_state: int = 0,
) -> ForecastEvaluation:
    import time

    models = models or list(MODEL_REGISTRY)
    rows = build_supervised(demand_wide, horizon, lookback=config.lookback_days)
    dates = split_dates(pd.Timestamp(demand_wide.index.max()), config)
    train = rows[rows["target_date"] < dates["val_start"]]
    val = rows[(rows["target_date"] >= dates["val_start"]) & (rows["target_date"] < dates["test_start"])]
    test = rows[rows["target_date"] >= dates["test_start"]]
    if train.empty or val.empty or test.empty:
        raise ValueError("history too short for the configured train/validation/test split")

    val_records, test_records, zone_records, horizon_records = [], [], [], []
    quantiles, coverage, runtimes, residuals, test_residuals = {}, {}, {}, {}, {}
    test_preds = test[["zone_id", "origin_date", "target_date", "horizon", "y"]].copy()
    lo_q, hi_q = config.interval_lower_quantile, config.interval_upper_quantile

    for name in models:
        start = time.perf_counter()
        model = make_model(name, random_state).fit(train)
        val_pred = val.assign(pred=model.predict(val))
        val_records.append({"model": name, **regression_metrics(val_pred["y"], val_pred["pred"])})
        quantiles[name] = _interval_quantiles(val_pred, horizon, lo_q, hi_q)
        residuals[name] = _total_residual_matrix(val_pred, horizon, list(demand_wide.columns))

        model = make_model(name, random_state).fit(pd.concat([train, val]))
        pred = model.predict(test)
        test_preds[name] = pred
        test_records.append({"model": name, **regression_metrics(test["y"], pred)})
        lo, hi = quantiles[name]["daily"]
        lower = np.maximum(pred * (1 + lo), 0.0)
        upper = pred * (1 + hi)
        y = test["y"].to_numpy()
        coverage[name] = float(np.mean((y >= lower) & (y <= upper)))
        frame = test.assign(pred=pred)
        test_residuals[name] = _total_residual_matrix(frame, horizon, list(demand_wide.columns))
        for zone_id, grp in frame.groupby("zone_id"):
            zone_records.append({"model": name, "zone_id": zone_id, **regression_metrics(grp["y"], grp["pred"])})
        for h, grp in frame.groupby("horizon"):
            horizon_records.append({"model": name, "horizon": int(h), **regression_metrics(grp["y"], grp["pred"])})
        runtimes[name] = time.perf_counter() - start

    val_df = pd.DataFrame(val_records)
    selected = str(val_df.sort_values(["mae", "model"]).iloc[0]["model"])
    return ForecastEvaluation(
        horizon=horizon,
        split={k: str(v.date()) for k, v in dates.items()}
        | {"train_end": str((dates["val_start"] - pd.Timedelta(days=1)).date())},
        validation_metrics=val_df,
        test_metrics=pd.DataFrame(test_records),
        test_per_zone=pd.DataFrame(zone_records),
        test_per_horizon=pd.DataFrame(horizon_records),
        test_predictions=test_preds.reset_index(drop=True),
        interval_quantiles=quantiles,
        test_interval_coverage=coverage,
        selected_model=selected,
        runtime_seconds=runtimes,
        validation_total_residuals=residuals,
        test_total_residuals=test_residuals,
    )


def forecast_future(
    demand_wide: pd.DataFrame,
    model_name: str,
    horizon: int,
    quantiles: dict[str, tuple[float, float]],
    lookback: int = 28,
    random_state: int = 0,
) -> ZoneForecast:
    """Fit on all observed history and forecast the next `horizon` days per zone."""
    rows = build_supervised(demand_wide, horizon, lookback=lookback, include_future=True)
    labelled = rows.dropna(subset=["y"])
    origin = pd.Timestamp(demand_wide.index.max())
    future = rows[rows["origin_date"] == origin].copy()
    model = make_model(model_name, random_state).fit(labelled)
    pred = np.maximum(model.predict(future), 0.0)
    lo, hi = quantiles.get("daily", (0.0, 0.0))
    future["predicted"] = pred
    future["lower"] = np.maximum(pred * (1 + lo), 0.0)
    future["upper"] = np.maximum(pred * (1 + hi), 0.0)
    future["model"] = model_name
    daily = future[["zone_id", "target_date", "horizon", "predicted", "lower", "upper", "model"]]
    daily = daily.sort_values(["zone_id", "horizon"]).reset_index(drop=True)

    lo_t, hi_t = quantiles.get("total", (lo, hi))
    totals = daily.groupby("zone_id", sort=False)["predicted"].sum().reindex(demand_wide.columns)
    totals = totals.rename("forecast_total").reset_index()
    totals["lower_total"] = np.maximum(totals["forecast_total"] * (1 + lo_t), 0.0)
    totals["upper_total"] = np.maximum(totals["forecast_total"] * (1 + hi_t), 0.0)
    totals["model"] = model_name
    note = (
        "Empirical interval: validation relative-residual quantiles "
        f"({lo:+.1%} / {hi:+.1%} daily, {lo_t:+.1%} / {hi_t:+.1%} over the horizon). "
        "Not a guaranteed probabilistic interval."
    )
    return ZoneForecast(model=model_name, origin_date=origin, daily=daily, totals=totals, interval_note=note)
