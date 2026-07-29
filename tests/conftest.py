"""Shared pytest fixtures.

``scripts/`` is not an installed package, so tests import from it by path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def daily_series() -> pd.DataFrame:
    """A clean 200-day single-SKU series with a weekly cycle and a mild trend."""
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    index = np.arange(len(dates), dtype=float)
    weekly = 10.0 * np.sin(2 * np.pi * index / 7.0)
    units = 100.0 + 0.2 * index + weekly
    return pd.DataFrame({"date": dates, "sku": "SKU-TEST", "units": units})


@pytest.fixture
def promotion_series() -> pd.DataFrame:
    """A 100-day single-SKU series with two promotion windows.

    Promotions run on days 20-25 (indices 20-25) and days 60-65.
    The last day (index 99) is a promotion day so look-ahead tests can
    perturb ``units`` there and check that no earlier feature moves.
    """
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    index = np.arange(100, dtype=float)
    units = 100.0 + 0.2 * index
    on_promotion = np.zeros(100, dtype=bool)
    on_promotion[20:26] = True  # first window
    on_promotion[60:66] = True  # second window
    on_promotion[99] = True  # final row is a promo day for the look-ahead test
    return pd.DataFrame(
        {"date": dates, "sku": "SKU-TEST", "units": units, "on_promotion": on_promotion}
    )


@pytest.fixture
def multi_sku_frame() -> pd.DataFrame:
    """Two overlapping SKU series, for aggregation and selection tests."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    frames = [
        pd.DataFrame({"date": dates, "sku": "SKU-A", "units": np.arange(60, dtype=float)}),
        pd.DataFrame({"date": dates, "sku": "SKU-B", "units": np.full(60, 5.0)}),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def metrics_payload() -> dict:
    """A minimal but well-formed reports/metrics.json payload."""
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "dataset": {"rows": 100, "skus": ["SKU-A"], "start": "2023-01-01", "end": "2023-04-10"},
        "config": {
            "series": "TOTAL",
            "horizon": 14,
            "n_splits": 5,
            "baseline_model": "seasonal_naive",
            "champion_model": "sarimax",
        },
        "models": {
            "seasonal_naive": {
                "model": "seasonal_naive",
                "params": {},
                "n_splits": 5,
                "horizon": 14,
                "metrics": {
                    "mae": 90.0,
                    "rmse": 110.0,
                    "mape": 0.10,
                    "smape": 0.10,
                    "bias": -10.0,
                },
            },
            "sarimax": {
                "model": "sarimax",
                "params": {},
                "n_splits": 5,
                "horizon": 14,
                "metrics": {
                    "mae": 80.0,
                    "rmse": 100.0,
                    "mape": 0.09,
                    "smape": 0.09,
                    "bias": -5.0,
                },
            },
        },
    }


@pytest.fixture
def thresholds() -> dict:
    return {
        "max_mape": 0.11,
        "max_mae": 95.0,
        "max_rmse": 120.0,
        "must_beat_baseline": True,
        "min_relative_improvement": 0.02,
        "max_abs_bias": 45.0,
        "min_splits": 5,
    }
