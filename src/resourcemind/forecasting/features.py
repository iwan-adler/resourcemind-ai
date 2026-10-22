"""Leakage-safe supervised features for direct multi-horizon forecasting.

Each row is (zone, forecast origin t, horizon h) with target y[t + h].
Every feature uses only observations at or before the origin t:

* mean_7, mean_28, std_28: rolling statistics over days (t-6..t) and (t-27..t)
* last_value: y[t]
* lag_dow_1/2/3: same weekday as the target, 1/2/3 weeks before it
  (y[t+h-7], y[t+h-14], y[t+h-21]); valid because h <= 7 implies t+h-7 <= t
* target calendar: weekday, annual sine/cosine (known in advance)
* horizon h and zone index (categorical)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_HORIZON = 7
MIN_HISTORY_DAYS = 29  # 28-day lookback + at least one origin
FEATURE_COLUMNS = [
    "zone_index",
    "horizon",
    "target_dow",
    "doy_sin",
    "doy_cos",
    "last_value",
    "mean_7",
    "mean_28",
    "std_28",
    "lag_dow_1",
    "lag_dow_2",
    "lag_dow_3",
]


class InsufficientHistoryError(ValueError):
    """Raised when the history is too short to build leakage-safe features."""


def _rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean and std over `window` rows (inclusive of the current row)."""
    n = values.shape[0]
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    csq = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values**2, axis=0)])
    mean = np.full_like(values, np.nan, dtype=float)
    std = np.full_like(values, np.nan, dtype=float)
    for t in range(window - 1, n):
        s = csum[t + 1] - csum[t + 1 - window]
        sq = csq[t + 1] - csq[t + 1 - window]
        mean[t] = s / window
        std[t] = np.sqrt(np.maximum(sq / window - mean[t] ** 2, 0.0))
    return mean, std


def build_supervised(
    demand_wide: pd.DataFrame,
    horizon: int,
    lookback: int = 28,
    include_future: bool = False,
) -> pd.DataFrame:
    """Build one row per (zone, origin, horizon).

    Rows whose target lies beyond the last observed date are only included when
    ``include_future`` is True (their ``y`` is NaN); those rows are used for
    out-of-sample forecasts.
    """
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON}")
    n_days, n_zones = demand_wide.shape
    if n_days < max(lookback, 21) + 1:
        raise InsufficientHistoryError(f"need at least {max(lookback, 21) + 1} days of history, got {n_days}")
    values = demand_wide.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(demand_wide.index)
    zone_ids = list(demand_wide.columns)
    mean7, _ = _rolling(values, 7)
    mean28, std28 = _rolling(values, lookback)

    frames = []
    first_origin = max(lookback, 21) - 1
    last_origin = n_days - 1 if include_future else n_days - 2
    origins = np.arange(first_origin, last_origin + 1)
    for h in range(1, horizon + 1):
        t = origins
        target_idx = t + h
        observed = target_idx < n_days
        if not include_future:
            t, target_idx = t[observed], target_idx[observed]
            observed = observed[observed]
        target_dates = dates[t] + pd.Timedelta(days=h)
        doy = target_dates.dayofyear.to_numpy()
        for z in range(n_zones):
            y = np.full(len(t), np.nan)
            y[observed] = values[target_idx[observed], z]
            frames.append(
                pd.DataFrame(
                    {
                        "zone_id": zone_ids[z],
                        "zone_index": z,
                        "origin_date": dates[t],
                        "target_date": target_dates,
                        "horizon": h,
                        "target_dow": target_dates.dayofweek.to_numpy(),
                        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                        "last_value": values[t, z],
                        "mean_7": mean7[t, z],
                        "mean_28": mean28[t, z],
                        "std_28": std28[t, z],
                        "lag_dow_1": values[t + h - 7, z],
                        "lag_dow_2": values[t + h - 14, z],
                        "lag_dow_3": values[t + h - 21, z],
                        "y": y,
                    }
                )
            )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["origin_date", "zone_index", "horizon"]).reset_index(drop=True)
