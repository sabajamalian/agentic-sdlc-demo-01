"""SARIMAX forecaster backed by statsmodels."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.models.base import Forecaster


class SarimaxForecaster(Forecaster):
    """Seasonal ARIMA on the raw target series.

    Defaults are tuned for daily data with a weekly cycle and deliberately kept
    small so the backtest stays fast enough to run on every pull request.
    """

    name = "sarimax"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
        trend: str | None = None,
        maxiter: int = 50,
    ) -> None:
        super().__init__(order=order, seasonal_order=seasonal_order, trend=trend, maxiter=maxiter)
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.maxiter = maxiter
        self._result = None

    def fit(self, history: pd.DataFrame) -> SarimaxForecaster:
        values = self._extract_target(history)
        min_required = max(2 * self.seasonal_order[3], 10)
        if values.size < min_required:
            raise ValueError(
                f"Need at least {min_required} observations to fit {self.name}, got {values.size}"
            )

        model = SARIMAX(
            values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        # statsmodels is noisy about convergence on short windows; the backtest
        # already reports whether the fit was any good.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit(disp=False, maxiter=self.maxiter)

        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        horizon = self._check_horizon(horizon)
        assert self._result is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._result.forecast(steps=horizon)
        return np.asarray(forecast, dtype=float)
