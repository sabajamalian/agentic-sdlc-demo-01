"""Feature engineering for daily demand series.

Every derived feature here is either a property of the timestamp itself
(deterministic, known in advance) or an explicitly lagged value of the target.
Nothing reads the target at time ``t`` to build a feature for time ``t``. That
rule is what ``tests/test_features.py`` enforces, and any agent-authored feature
is expected to keep it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.data import DATE_COLUMN, TARGET_COLUMN

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 28)


def add_calendar_features(frame: pd.DataFrame, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Add features derived purely from the timestamp.

    These are safe at any horizon because a future date's day-of-week is known
    today.
    """
    out = frame.copy()
    dates = pd.to_datetime(out[date_column])

    out["day_of_week"] = dates.dt.dayofweek.astype("int16")
    out["day_of_month"] = dates.dt.day.astype("int16")
    out["day_of_year"] = dates.dt.dayofyear.astype("int16")
    out["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    out["month"] = dates.dt.month.astype("int16")
    out["quarter"] = dates.dt.quarter.astype("int16")
    out["year"] = dates.dt.year.astype("int16")
    out["is_weekend"] = dates.dt.dayofweek.isin((5, 6))
    out["is_month_start"] = dates.dt.is_month_start
    out["is_month_end"] = dates.dt.is_month_end

    return out


def add_fourier_terms(
    frame: pd.DataFrame,
    period: float = 365.25,
    order: int = 3,
    date_column: str = DATE_COLUMN,
    prefix: str = "yearly",
) -> pd.DataFrame:
    """Add sine/cosine pairs describing a seasonal cycle of ``period`` days."""
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")

    out = frame.copy()
    dates = pd.to_datetime(out[date_column])
    elapsed = (dates - dates.min()).dt.days.to_numpy(dtype=float)

    for k in range(1, order + 1):
        angle = 2.0 * np.pi * k * elapsed / period
        out[f"{prefix}_sin_{k}"] = np.sin(angle)
        out[f"{prefix}_cos_{k}"] = np.cos(angle)

    return out


def add_lag_features(
    frame: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add ``target`` shifted back by each lag.

    ``lag_k`` at row ``t`` holds the target from row ``t - k``, so the smallest
    lag decides the shortest horizon the resulting model can honestly forecast.
    """
    if any(lag < 1 for lag in lags):
        raise ValueError(f"All lags must be >= 1 to avoid leaking the present, got {lags}")

    out = frame.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_column].shift(lag)
    return out


def add_rolling_features(
    frame: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    target_column: str = TARGET_COLUMN,
    min_lag: int = 1,
) -> pd.DataFrame:
    """Add rolling mean and std over windows that end ``min_lag`` steps in the past.

    The target is shifted before the window is applied, so the current
    observation is never part of its own summary statistic.
    """
    if min_lag < 1:
        raise ValueError(f"min_lag must be >= 1 to avoid leaking the present, got {min_lag}")
    if any(window < 2 for window in windows):
        raise ValueError(f"All rolling windows must be >= 2, got {windows}")

    out = frame.copy()
    shifted = out[target_column].shift(min_lag)
    for window in windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window, min_periods=window).std()
    return out


def build_feature_frame(
    frame: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    fourier_order: int = 3,
    dropna: bool = True,
) -> pd.DataFrame:
    """Run the standard feature pipeline over a single sorted series."""
    out = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    out = add_calendar_features(out)
    out = add_fourier_terms(out, order=fourier_order)
    out = add_lag_features(out, lags=lags)
    out = add_rolling_features(out, windows=rolling_windows)
    return out.dropna().reset_index(drop=True) if dropna else out


def feature_columns(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    """Numeric/boolean columns usable as model inputs."""
    excluded = {DATE_COLUMN, target_column, "sku"}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
