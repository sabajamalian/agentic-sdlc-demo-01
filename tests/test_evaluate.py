"""Metric correctness and backtest behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.data import aggregate_total
from forecasting.evaluate import (
    METRICS,
    backtest,
    bias,
    compare_models,
    mae,
    mape,
    per_sku_metrics,
    rmse,
    score,
    smape,
)


class TestMetricValues:
    """Hand-computed values, so a refactor that changes the maths is caught."""

    def test_mae(self):
        assert mae([10.0, 20.0, 30.0], [12.0, 18.0, 33.0]) == pytest.approx((2 + 2 + 3) / 3)

    def test_rmse(self):
        assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))

    def test_mape(self):
        # errors of 10% and 20%
        assert mape([100.0, 50.0], [110.0, 40.0]) == pytest.approx(0.15)

    def test_smape(self):
        # |100-110| / ((100+110)/2) = 10/105
        assert smape([100.0], [110.0]) == pytest.approx(10 / 105)

    def test_bias_is_signed(self):
        assert bias([10.0, 10.0], [12.0, 14.0]) == pytest.approx(3.0)
        assert bias([10.0, 10.0], [8.0, 6.0]) == pytest.approx(-3.0)

    def test_perfect_forecast_scores_zero(self):
        actual = [1.0, 2.0, 3.0]
        for name, fn in METRICS.items():
            assert fn(actual, actual) == pytest.approx(0.0), name


class TestMetricGuards:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            mae([1.0, 2.0], [1.0])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mae([], [])

    def test_mape_skips_zero_actuals_rather_than_dividing_by_zero(self):
        # The zero actual is dropped, leaving a single 10% error.
        assert mape([0.0, 100.0], [5.0, 110.0]) == pytest.approx(0.10)

    def test_mape_all_zero_actuals_raises(self):
        with pytest.raises(ValueError, match="undefined"):
            mape([0.0, 0.0], [1.0, 2.0])

    def test_smape_all_zero_raises(self):
        with pytest.raises(ValueError, match="undefined"):
            smape([0.0, 0.0], [0.0, 0.0])

    def test_smape_is_bounded_by_two(self):
        assert smape([1.0], [-1.0]) <= 2.0 + 1e-12


class TestScore:
    def test_returns_every_registered_metric(self):
        result = score([1.0, 2.0], [1.5, 2.5])
        assert set(result) == set(METRICS)
        assert all(isinstance(value, float) for value in result.values())


class TestBacktest:
    def test_runs_the_requested_number_of_origins(self, daily_series):
        result = backtest(daily_series, model="seasonal_naive", horizon=14, n_splits=5)
        assert result.n_splits == 5
        assert len(result.folds) == 5

    def test_reports_pooled_metrics(self, daily_series):
        result = backtest(daily_series, model="seasonal_naive", horizon=14, n_splits=3)
        assert set(result.metrics) == set(METRICS)
        assert result.metrics["mape"] > 0

    def test_fold_metadata_is_populated(self, daily_series):
        result = backtest(daily_series, model="mean", horizon=7, n_splits=3)
        for index, fold in enumerate(result.folds):
            assert fold.fold == index
            assert fold.horizon == 7
            assert fold.train_size > 0
            assert len(fold.cutoff) == 10  # YYYY-MM-DD

    def test_training_windows_grow_across_folds(self, daily_series):
        result = backtest(daily_series, model="mean", horizon=7, n_splits=4)
        sizes = [fold.train_size for fold in result.folds]
        assert sizes == sorted(sizes)

    def test_collect_predictions_returns_aligned_rows(self, daily_series):
        result = backtest(
            daily_series, model="mean", horizon=7, n_splits=3, collect_predictions=True
        )
        assert result.predictions is not None
        assert len(result.predictions) == 21
        assert set(result.predictions.columns) == {"fold", "date", "actual", "predicted"}

    def test_predictions_are_omitted_by_default(self, daily_series):
        assert backtest(daily_series, model="mean", horizon=7, n_splits=2).predictions is None

    def test_model_params_reach_the_forecaster(self, daily_series):
        result = backtest(
            daily_series,
            model="seasonal_naive",
            horizon=7,
            n_splits=2,
            model_params={"season_length": 14},
        )
        assert result.params == {"season_length": 14}

    def test_perfect_model_would_score_zero(self, daily_series):
        """A flat series is exactly reproduced by seasonal naive, so error is zero."""
        flat = daily_series.copy()
        flat["units"] = 100.0
        result = backtest(flat, model="seasonal_naive", horizon=7, n_splits=3)
        assert result.metrics["mape"] == pytest.approx(0.0)

    def test_to_dict_is_json_serialisable(self, daily_series):
        import json

        result = backtest(daily_series, model="mean", horizon=7, n_splits=2)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["model"] == "mean"
        assert len(payload["folds"]) == 2

    def test_to_dict_can_omit_folds(self, daily_series):
        result = backtest(daily_series, model="mean", horizon=7, n_splits=2)
        assert "folds" not in result.to_dict(include_folds=False)

    def test_unknown_model_raises(self, daily_series):
        with pytest.raises(KeyError, match="Unknown model"):
            backtest(daily_series, model="does_not_exist", horizon=7, n_splits=2)


class TestCompareModels:
    def test_returns_one_row_per_model_sorted_by_mape(self, daily_series):
        table = compare_models(
            daily_series, {"seasonal_naive": {}, "mean": {}}, horizon=7, n_splits=3
        )
        assert len(table) == 2
        assert list(table["model"])
        assert table["mape"].is_monotonic_increasing

    def test_includes_every_metric_column(self, daily_series):
        table = compare_models(daily_series, {"mean": {}}, horizon=7, n_splits=2)
        assert set(METRICS) <= set(table.columns)


class TestPerSkuMetrics:
    def test_sorts_worst_first_and_differs_from_aggregate(self):
        dates = pd.date_range("2024-01-01", periods=28, freq="D")
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "sku": "SKU-STABLE",
                        "units": np.full(len(dates), 100.0),
                    }
                ),
                pd.DataFrame(
                    {
                        "date": dates,
                        "sku": "SKU-SWING",
                        "units": np.where(np.arange(len(dates)) % 2 == 0, 10.0, 200.0),
                    }
                ),
            ],
            ignore_index=True,
        )

        by_sku = per_sku_metrics(frame, model="mean", horizon=7, n_splits=2, model_params={"window": 7})
        total = backtest(
            aggregate_total(frame),
            model="mean",
            horizon=7,
            n_splits=2,
            model_params={"window": 7},
        )

        assert list(by_sku["sku"]) == ["SKU-SWING", "SKU-STABLE"]
        assert by_sku["mape"].is_monotonic_decreasing
        assert any(not np.isclose(value, total.metrics["mape"]) for value in by_sku["mape"])
