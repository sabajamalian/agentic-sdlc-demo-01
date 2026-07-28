"""Demand forecasting toolkit for the agentic SDLC demo.

The public surface is intentionally small so that agent-authored changes have an
obvious place to land:

* add a model  -> ``forecasting.models`` + one line in ``forecasting.registry``
* add features -> ``forecasting.features``
* add a metric -> ``forecasting.evaluate``
"""

from forecasting.data import load_sales, rolling_origin_splits, train_test_split
from forecasting.evaluate import backtest, mae, mape, smape
from forecasting.registry import available_models, get_model

__all__ = [
    "available_models",
    "backtest",
    "get_model",
    "load_sales",
    "mae",
    "mape",
    "rolling_origin_splits",
    "smape",
    "train_test_split",
]
