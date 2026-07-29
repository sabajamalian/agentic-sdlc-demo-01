"""Accuracy metrics and the rolling-origin backtest.

The backtest re-fits the model at every origin. That is slower than fitting once
and slicing, but it is the only version that answers the question the eval gate
actually asks: how would this model have performed if we had deployed it?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from forecasting.data import (
    DATE_COLUMN,
    SKU_COLUMN,
    TARGET_COLUMN,
    Split,
    rolling_origin_splits,
    select_sku,
)
from forecasting.models.base import Forecaster
from forecasting.registry import get_model

EPSILON = 1e-8


def _as_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}")
    if truth.size == 0:
        raise ValueError("Cannot compute a metric over an empty array")
    return truth, pred


def mae(y_true: object, y_pred: object) -> float:
    """Mean absolute error."""
    truth, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred)))


def rmse(y_true: object, y_pred: object) -> float:
    """Root mean squared error."""
    truth, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mape(y_true: object, y_pred: object) -> float:
    """Mean absolute percentage error, as a fraction (0.12 means 12%).

    Zero actuals are dropped rather than clamped, because clamping quietly
    invents an error value that nobody can interpret.
    """
    truth, pred = _as_arrays(y_true, y_pred)
    mask = np.abs(truth) > EPSILON
    if not mask.any():
        raise ValueError("MAPE is undefined when every actual value is zero")
    return float(np.mean(np.abs((truth[mask] - pred[mask]) / truth[mask])))


def smape(y_true: object, y_pred: object) -> float:
    """Symmetric MAPE as a fraction, bounded in [0, 2]."""
    truth, pred = _as_arrays(y_true, y_pred)
    denominator = (np.abs(truth) + np.abs(pred)) / 2.0
    mask = denominator > EPSILON
    if not mask.any():
        raise ValueError("sMAPE is undefined when every actual and prediction is zero")
    return float(np.mean(np.abs(truth[mask] - pred[mask]) / denominator[mask]))


def bias(y_true: object, y_pred: object) -> float:
    """Mean signed error. Positive means the model is over-forecasting."""
    truth, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(pred - truth))


METRICS = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
    "bias": bias,
}


def score(y_true: object, y_pred: object) -> dict[str, float]:
    """Compute every registered metric at once."""
    return {name: fn(y_true, y_pred) for name, fn in METRICS.items()}


@dataclass
class FoldResult:
    """Metrics for one origin of the backtest."""

    fold: int
    cutoff: str
    train_size: int
    horizon: int
    metrics: dict[str, float]


@dataclass
class BacktestResult:
    """Aggregate outcome of a rolling-origin backtest."""

    model: str
    params: dict[str, object]
    n_splits: int
    horizon: int
    metrics: dict[str, float]
    folds: list[FoldResult] = field(default_factory=list)
    predictions: pd.DataFrame | None = None

    def to_dict(self, include_folds: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "params": self.params,
            "n_splits": self.n_splits,
            "horizon": self.horizon,
            "metrics": self.metrics,
        }
        if include_folds:
            payload["folds"] = [asdict(fold) for fold in self.folds]
        return payload


def backtest(
    frame: pd.DataFrame,
    model: str | Forecaster = "seasonal_naive",
    horizon: int = 14,
    n_splits: int = 5,
    step: int | None = None,
    model_params: dict[str, object] | None = None,
    collect_predictions: bool = False,
) -> BacktestResult:
    """Run a rolling-origin backtest over a single series.

    ``model`` may be a registry name or a Forecaster instance. When a name is
    given, a fresh instance is built for every fold so no state leaks between
    origins.
    """
    model_params = dict(model_params or {})
    model_name = model if isinstance(model, str) else type(model).name

    fold_results: list[FoldResult] = []
    prediction_rows: list[pd.DataFrame] = []
    all_truth: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    splits: list[Split] = list(
        rolling_origin_splits(frame, horizon=horizon, n_splits=n_splits, step=step)
    )

    for index, split in enumerate(splits):
        forecaster = get_model(model_name, **model_params) if isinstance(model, str) else model
        forecaster.fit(split.train)
        predicted = forecaster.predict(split.horizon)

        actual = split.test[TARGET_COLUMN].to_numpy(dtype=float)
        all_truth.append(actual)
        all_pred.append(predicted)

        fold_results.append(
            FoldResult(
                fold=index,
                cutoff=str(pd.Timestamp(split.cutoff).date()),
                train_size=len(split.train),
                horizon=split.horizon,
                metrics=score(actual, predicted),
            )
        )

        if collect_predictions:
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": index,
                        DATE_COLUMN: split.test[DATE_COLUMN].to_numpy(),
                        "actual": actual,
                        "predicted": predicted,
                    }
                )
            )

    pooled_truth = np.concatenate(all_truth)
    pooled_pred = np.concatenate(all_pred)

    return BacktestResult(
        model=model_name,
        params=model_params,
        n_splits=len(splits),
        horizon=horizon,
        metrics=score(pooled_truth, pooled_pred),
        folds=fold_results,
        predictions=pd.concat(prediction_rows, ignore_index=True) if prediction_rows else None,
    )


def compare_models(
    frame: pd.DataFrame,
    models: dict[str, dict[str, object]],
    horizon: int = 14,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Backtest several models and return a metrics table sorted by MAPE."""
    rows = []
    for name, params in models.items():
        result = backtest(
            frame, model=name, horizon=horizon, n_splits=n_splits, model_params=params
        )
        rows.append({"model": name, **result.metrics})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def per_sku_metrics(
    frame: pd.DataFrame,
    model: str = "seasonal_naive",
    horizon: int = 14,
    n_splits: int = 5,
    model_params: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Backtest one model per SKU and return a worst-first MAPE table."""
    rows: list[dict[str, object]] = []
    for sku in sorted(frame[SKU_COLUMN].unique()):
        result = backtest(
            select_sku(frame, sku),
            model=model,
            horizon=horizon,
            n_splits=n_splits,
            model_params=model_params,
        )
        rows.append({"sku": sku, **result.metrics})
    return pd.DataFrame(rows).sort_values("mape", ascending=False).reset_index(drop=True)
