"""Baseline forecasters.

The seasonal naive model is the bar every other model has to clear. The eval
gate in ``scripts/check_eval_gate.py`` compares against it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.models.base import Forecaster


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the last full season of observations.

    With ``season_length=7`` on daily data, next Tuesday is forecast as last
    Tuesday. Cheap, hard to beat on short horizons, and a genuine sanity check
    on anything more complicated.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 7) -> None:
        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")
        super().__init__(season_length=season_length)
        self.season_length = season_length
        self._season: np.ndarray | None = None

    def fit(self, history: pd.DataFrame) -> SeasonalNaiveForecaster:
        values = self._extract_target(history)
        if values.size < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations to fit "
                f"{self.name}, got {values.size}"
            )
        self._season = values[-self.season_length :].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        horizon = self._check_horizon(horizon)
        assert self._season is not None
        repeats = int(np.ceil(horizon / self.season_length))
        return np.tile(self._season, repeats)[:horizon]


class MeanForecaster(Forecaster):
    """Forecast the mean of the last ``window`` observations (or all of them)."""

    name = "mean"

    def __init__(self, window: int | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1 when set, got {window}")
        super().__init__(window=window)
        self.window = window
        self._value: float | None = None

    def fit(self, history: pd.DataFrame) -> MeanForecaster:
        values = self._extract_target(history)
        tail = values if self.window is None else values[-self.window :]
        self._value = float(np.mean(tail))
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        horizon = self._check_horizon(horizon)
        assert self._value is not None
        return np.full(horizon, self._value, dtype=float)
