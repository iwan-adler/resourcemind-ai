"""Forecasting models sharing one fit/predict interface over supervised rows.

Baselines need no fitting; they read leakage-safe columns built in features.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from resourcemind.forecasting.features import FEATURE_COLUMNS

if TYPE_CHECKING:
    from sklearn.ensemble import HistGradientBoostingRegressor


class Forecaster:
    name: str = "base"
    label: str = "Base"

    def fit(self, rows: pd.DataFrame) -> Forecaster:
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class SeasonalNaive(Forecaster):
    """Demand on the same weekday one week earlier."""

    name, label = "seasonal_naive", "Seasonal naive (same weekday last week)"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["lag_dow_1"].to_numpy(dtype=float)


class SeasonalAverage(Forecaster):
    """Mean of the same weekday over the previous three weeks."""

    name, label = "seasonal_average_3w", "Seasonal average (same weekday, 3 weeks)"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[["lag_dow_1", "lag_dow_2", "lag_dow_3"]].mean(axis=1).to_numpy(dtype=float)


class MovingAverage(Forecaster):
    """Trailing 28-day mean (ignores weekly seasonality)."""

    name, label = "moving_average_28d", "Moving average (28 days)"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["mean_28"].to_numpy(dtype=float)


class GradientBoosting(Forecaster):
    """Global direct multi-horizon model: one regressor for all zones and horizons.

    Poisson loss keeps predictions strictly non-negative and suits count data.
    Early stopping is disabled so no random internal split is used.
    """

    name, label = "gradient_boosting", "Gradient boosting (lags + rolling + calendar)"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.model: HistGradientBoostingRegressor | None = None
        self.constant: float | None = None

    def fit(self, rows: pd.DataFrame) -> GradientBoosting:
        rows = rows.dropna(subset=["y"])
        y = rows["y"].to_numpy(dtype=float)
        if len(y) == 0:
            raise ValueError("no labelled rows to fit")
        # Poisson loss needs positive mean demand; an all-zero or constant
        # series is handled exactly by a constant predictor.
        if y.sum() <= 0 or np.allclose(y, y[0]):
            self.constant, self.model = float(y.mean()), None
            return self
        self.constant = None
        # Imported here: scikit-learn is slow to import and only needed when fitting.
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            categorical_features=[FEATURE_COLUMNS.index("zone_index")],
            random_state=self.random_state,
        )
        self.model.fit(rows[FEATURE_COLUMNS].to_numpy(dtype=float), y)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(rows), self.constant)
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model.predict(rows[FEATURE_COLUMNS].to_numpy(dtype=float))


MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    m.name: m for m in (SeasonalNaive, SeasonalAverage, MovingAverage, GradientBoosting)
}
BASELINE_MODELS = ["seasonal_naive", "seasonal_average_3w", "moving_average_28d"]


def make_model(name: str, random_state: int = 0) -> Forecaster:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    return cls(random_state=random_state) if cls is GradientBoosting else cls()


def model_label(name: str) -> str:
    return MODEL_REGISTRY[name].label
