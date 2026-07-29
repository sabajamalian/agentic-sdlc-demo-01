"""Loading and splitting of the sales dataset.

Every split here is chronological. Random splits leak the future into the
training set for a time series, so they are deliberately not offered.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sales.csv"

DATE_COLUMN = "date"
TARGET_COLUMN = "units"
SKU_COLUMN = "sku"


@dataclass(frozen=True)
class Split:
    """One chronological train/test partition of a series."""

    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def horizon(self) -> int:
        return len(self.test)

    @property
    def cutoff(self) -> pd.Timestamp:
        """Last timestamp available to the model at training time."""
        return self.train[DATE_COLUMN].max()


def load_sales(path: str | Path | None = None) -> pd.DataFrame:
    """Load the sales dataset, sorted by SKU then date.

    Returns a frame with columns ``date`` (datetime64), ``sku`` (str),
    ``units`` (float), ``price`` (float) and ``on_promotion`` (bool).
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Sales data not found at {path}. Run `make data` to generate it.")

    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    missing = {DATE_COLUMN, SKU_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Sales data at {path} is missing columns: {sorted(missing)}")

    frame[SKU_COLUMN] = frame[SKU_COLUMN].astype(str)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(float)
    if "on_promotion" in frame.columns:
        frame["on_promotion"] = frame["on_promotion"].astype(bool)

    return frame.sort_values([SKU_COLUMN, DATE_COLUMN]).reset_index(drop=True)


def select_sku(frame: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Return the single-SKU series for ``sku``, sorted by date."""
    subset = frame.loc[frame[SKU_COLUMN] == sku]
    if subset.empty:
        known = sorted(frame[SKU_COLUMN].unique())
        raise KeyError(f"Unknown sku {sku!r}. Available: {known}")
    return subset.sort_values(DATE_COLUMN).reset_index(drop=True)


def aggregate_total(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse all SKUs into a single total-demand series."""
    total = (
        frame.groupby(DATE_COLUMN, as_index=False)[TARGET_COLUMN]
        .sum()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    total[SKU_COLUMN] = "TOTAL"
    return total


def train_test_split(frame: pd.DataFrame, horizon: int) -> Split:
    """Hold out the final ``horizon`` rows as the test set."""
    validate_single_series(frame)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if horizon >= len(frame):
        raise ValueError(
            f"horizon {horizon} leaves no training data for a series of length {len(frame)}"
        )

    ordered = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    return Split(
        train=ordered.iloc[:-horizon].reset_index(drop=True),
        test=ordered.iloc[-horizon:].reset_index(drop=True),
    )


def rolling_origin_splits(
    frame: pd.DataFrame,
    horizon: int,
    n_splits: int,
    step: int | None = None,
    min_train_size: int | None = None,
) -> Iterator[Split]:
    """Yield expanding-window splits, oldest origin first.

    Each split trains on everything up to a cutoff and tests on the next
    ``horizon`` observations. Windows never overlap the future, which is what
    makes the resulting metrics honest.
    """
    validate_single_series(frame)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if n_splits <= 0:
        raise ValueError(f"n_splits must be positive, got {n_splits}")

    step = horizon if step is None else step
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    ordered = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    total = len(ordered)

    # Origin of the final (most recent) split, then walk backwards by `step`.
    last_train_end = total - horizon
    first_train_end = last_train_end - (n_splits - 1) * step

    floor = min_train_size if min_train_size is not None else horizon
    if first_train_end < floor:
        needed = floor + (n_splits - 1) * step + horizon
        raise ValueError(
            f"Series of length {total} is too short for {n_splits} splits of horizon "
            f"{horizon} with step {step} (needs at least {needed} rows)"
        )

    for i in range(n_splits):
        train_end = first_train_end + i * step
        yield Split(
            train=ordered.iloc[:train_end].reset_index(drop=True),
            test=ordered.iloc[train_end : train_end + horizon].reset_index(drop=True),
        )


def validate_single_series(frame: pd.DataFrame) -> None:
    """Raise unless ``frame`` is one series: one row per date, at most one SKU.

    Anything that shifts, rolls or splits along the time axis needs this. On a
    multi-SKU frame those operations run across interleaved series and quietly
    mix one SKU's target into another's features.
    """
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"frame is missing the {DATE_COLUMN!r} column")
    if SKU_COLUMN in frame.columns and frame[SKU_COLUMN].nunique() > 1:
        raise ValueError(
            "Expected a single series but found "
            f"{frame[SKU_COLUMN].nunique()} SKUs. Use select_sku() or aggregate_total() first."
        )
    if frame[DATE_COLUMN].duplicated().any():
        raise ValueError("frame contains duplicate dates for a single series")
