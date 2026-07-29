"""Name -> forecaster factory.

Adding a model to the project is a one-line change here. Scripts, notebooks and
the eval gate all resolve models through this registry, so nothing else has to
learn about the new class.
"""

from __future__ import annotations

from collections.abc import Callable

from forecasting.models.base import Forecaster
from forecasting.models.naive import MeanForecaster, SeasonalNaiveForecaster
from forecasting.models.sarimax import SarimaxForecaster

ModelFactory = Callable[..., Forecaster]

_REGISTRY: dict[str, ModelFactory] = {
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
    MeanForecaster.name: MeanForecaster,
    SarimaxForecaster.name: SarimaxForecaster,
}

BASELINE_MODEL = SeasonalNaiveForecaster.name


def register_model(name: str, factory: ModelFactory, *, overwrite: bool = False) -> None:
    """Register a forecaster factory under ``name``."""
    if not name:
        raise ValueError("Model name must be a non-empty string")
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"Model {name!r} is already registered. Pass overwrite=True to replace it."
        )
    _REGISTRY[name] = factory


def get_model(name: str, **params: object) -> Forecaster:
    """Instantiate the forecaster registered under ``name``."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}") from None
    return factory(**params)


def available_models() -> list[str]:
    """Sorted list of registered model names."""
    return sorted(_REGISTRY)
