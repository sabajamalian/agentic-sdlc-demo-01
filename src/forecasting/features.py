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

from forecasting.data import DATE_COLUMN, TARGET_COLUMN, validate_single_series

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 28)

#: Fixed phase origin for :func:`add_fourier_terms`. Any constant date works;
#: what matters is that it does not depend on the frame being encoded, so a
#: given calendar day always maps to the same point on the seasonal cycle.
FOURIER_EPOCH = pd.Timestamp("2000-01-01")


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
    """Add sine/cosine pairs describing a seasonal cycle of ``period`` days.

    The phase is anchored to :data:`FOURIER_EPOCH`, a fixed calendar date, not
    to the first row of ``frame``. Anchoring to the frame would make the
    encoding slice-dependent: the same calendar day would get different sin/cos
    values in different backtest folds, so coefficients learned on one fold
    would not mean the same thing on the next.
    """
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")

    out = frame.copy()
    dates = pd.to_datetime(out[date_column])
    elapsed = (dates - FOURIER_EPOCH).dt.days.to_numpy(dtype=float)

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

    Rejects multi-SKU frames: a bare shift over interleaved series would put
    another SKU's contemporaneous value in this SKU's lag column.
    """
    if any(lag < 1 for lag in lags):
        raise ValueError(f"All lags must be >= 1 to avoid leaking the present, got {lags}")

    validate_single_series(frame)

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

    Rejects multi-SKU frames for the same reason as :func:`add_lag_features`.
    """
    if min_lag < 1:
        raise ValueError(f"min_lag must be >= 1 to avoid leaking the present, got {min_lag}")
    if any(window < 2 for window in windows):
        raise ValueError(f"All rolling windows must be >= 2, got {windows}")

    validate_single_series(frame)

    out = frame.copy()
    shifted = out[target_column].shift(min_lag)
    for window in windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window, min_periods=window).std()
    return out


def add_promotion_features(
    frame: pd.DataFrame,
    date_column: str = DATE_COLUMN,
) -> pd.DataFrame:
    """Add promotion-aware features derived from the on_promotion flag.

    Both outputs are safe at any forecast horizon: the promotion schedule is
    set by the business approximately six weeks in advance and is genuinely
    known at forecast time.

    ``on_promotion`` is cast to ``int8`` so tree and linear models use it
    directly.  ``days_until_next_promotion`` counts calendar days from each
    row to the nearest upcoming (or current) promotion date; it is ``NaN``
    when no later promotion exists in the frame.  Neither column reads
    ``units`` at any point, so there is no target leakage.

    Raises ``ValueError`` if ``on_promotion`` is not a column in ``frame``.
    """
    if "on_promotion" not in frame.columns:
        raise ValueError(
            "frame is missing the 'on_promotion' column required by add_promotion_features"
        )

    out = frame.copy()
    dates = pd.to_datetime(out[date_column])
    promo_flag = out["on_promotion"].astype(bool)

    # Cast to int8 so feature_columns() and all downstream models handle it uniformly.
    out["on_promotion"] = promo_flag.astype("int8")

    # days_until_next_promotion: derived only from the promotion schedule, not units.
    # 0 on a promotion day itself; NaN when no later promotion exists in the frame.
    promo_dates = dates[promo_flag].sort_values().to_numpy(dtype="datetime64[D]")
    date_arr = dates.to_numpy(dtype="datetime64[D]")

    if len(promo_dates) == 0:
        out["days_until_next_promotion"] = np.nan
    else:
        idx = np.searchsorted(promo_dates, date_arr, side="left")
        valid = idx < len(promo_dates)
        clipped = np.minimum(idx, len(promo_dates) - 1)
        diffs = (promo_dates[clipped] - date_arr).astype("timedelta64[D]").astype(float)
        out["days_until_next_promotion"] = np.where(valid, diffs, np.nan)

    return out


def build_feature_frame(
    frame: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    fourier_order: int = 3,
    dropna: bool = True,
) -> pd.DataFrame:
    """Run the standard feature pipeline over a single sorted series.

    Raises if handed more than one SKU. Aggregate with ``aggregate_total`` or
    pick one with ``select_sku`` first; to build features for every SKU, loop
    and call this once per series.
    """
    validate_single_series(frame)

    out = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    out = add_calendar_features(out)
    out = add_fourier_terms(out, order=fourier_order)
    out = add_lag_features(out, lags=lags)
    out = add_rolling_features(out, windows=rolling_windows)
    if "on_promotion" in out.columns:
        out = add_promotion_features(out)
    return out.dropna().reset_index(drop=True) if dropna else out


def feature_columns(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    """Numeric/boolean columns usable as model inputs."""
    excluded = {DATE_COLUMN, target_column, "sku"}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
