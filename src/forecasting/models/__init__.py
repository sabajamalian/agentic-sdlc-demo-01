"""Forecasting models.

Adding a model means creating a module here and registering it in
``forecasting.registry``. Nothing else in the codebase should need to change.
"""

from forecasting.models.base import Forecaster
from forecasting.models.naive import SeasonalNaiveForecaster
from forecasting.models.sarimax import SarimaxForecaster

__all__ = ["Forecaster", "SarimaxForecaster", "SeasonalNaiveForecaster"]
