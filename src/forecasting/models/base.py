"""The contract every forecaster in this project implements."""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd

from forecasting.data import DATE_COLUMN, TARGET_COLUMN


class Forecaster(abc.ABC):
    """Fit on a history frame, then predict the next ``horizon`` steps.

    Implementations must:

    * read only rows present in the frame passed to :meth:`fit`
    * return exactly ``horizon`` values from :meth:`predict`
    * be safe to re-fit on a longer history without carrying state forward
    """

    name: str = "forecaster"

    def __init__(self, **params: object) -> None:
        self.params = dict(params)
        self._fitted = False

    @abc.abstractmethod
    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit on a single sorted series containing ``date`` and ``units``."""

    @abc.abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return ``horizon`` point forecasts for the steps after the fitted history."""

    def fit_predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        return self.fit(history).predict(horizon)

    def __repr__(self) -> str:
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.params.items()))
        return f"{type(self).__name__}({rendered})"

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before calling predict()")

    @staticmethod
    def _extract_target(history: pd.DataFrame, target_column: str = TARGET_COLUMN) -> np.ndarray:
        if target_column not in history.columns:
            raise ValueError(f"history is missing the {target_column!r} column")
        if DATE_COLUMN in history.columns and not history[DATE_COLUMN].is_monotonic_increasing:
            raise ValueError("history must be sorted by date before fitting")
        values = history[target_column].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError("history is empty")
        if not np.isfinite(values).all():
            raise ValueError("history contains NaN or infinite target values")
        return values

    @staticmethod
    def _check_horizon(horizon: int) -> int:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        return horizon
