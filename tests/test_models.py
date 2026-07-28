"""Forecaster contract and registry tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.models import Forecaster, SarimaxForecaster, SeasonalNaiveForecaster
from forecasting.models.naive import MeanForecaster
from forecasting.registry import (
    BASELINE_MODEL,
    available_models,
    get_model,
    register_model,
)

CHEAP_MODELS = ["seasonal_naive", "mean"]


class TestSeasonalNaive:
    def test_repeats_the_last_season(self, daily_series):
        model = SeasonalNaiveForecaster(season_length=7).fit(daily_series)
        predicted = model.predict(7)
        expected = daily_series["units"].to_numpy()[-7:]
        np.testing.assert_allclose(predicted, expected)

    def test_tiles_beyond_one_season(self, daily_series):
        model = SeasonalNaiveForecaster(season_length=7).fit(daily_series)
        predicted = model.predict(10)
        assert len(predicted) == 10
        np.testing.assert_allclose(predicted[7:10], predicted[0:3])

    def test_rejects_history_shorter_than_a_season(self):
        short = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=3), "units": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(ValueError, match="at least 7 observations"):
            SeasonalNaiveForecaster(season_length=7).fit(short)

    def test_rejects_invalid_season_length(self):
        with pytest.raises(ValueError, match="season_length must be >= 1"):
            SeasonalNaiveForecaster(season_length=0)


class TestMeanForecaster:
    def test_uses_the_full_history_by_default(self, daily_series):
        model = MeanForecaster().fit(daily_series)
        assert model.predict(3)[0] == pytest.approx(daily_series["units"].mean())

    def test_window_limits_the_history_used(self, daily_series):
        model = MeanForecaster(window=10).fit(daily_series)
        assert model.predict(3)[0] == pytest.approx(daily_series["units"].tail(10).mean())

    def test_predictions_are_constant(self, daily_series):
        predicted = MeanForecaster().fit(daily_series).predict(5)
        assert len(set(predicted.tolist())) == 1


@pytest.mark.slow
class TestSarimax:
    def test_produces_the_requested_horizon(self, daily_series):
        predicted = SarimaxForecaster().fit(daily_series).predict(14)
        assert predicted.shape == (14,)
        assert np.isfinite(predicted).all()

    def test_beats_a_constant_forecast_on_a_seasonal_series(self, daily_series):
        train = daily_series.iloc[:-14]
        actual = daily_series["units"].to_numpy()[-14:]

        sarimax = SarimaxForecaster().fit(train).predict(14)
        constant = MeanForecaster().fit(train).predict(14)

        assert np.mean(np.abs(sarimax - actual)) < np.mean(np.abs(constant - actual))

    def test_rejects_history_that_is_too_short(self):
        short = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=8), "units": np.arange(8, dtype=float)}
        )
        with pytest.raises(ValueError, match="at least"):
            SarimaxForecaster().fit(short)


@pytest.mark.parametrize("name", CHEAP_MODELS)
class TestForecasterContract:
    def test_predict_before_fit_raises(self, name):
        with pytest.raises(RuntimeError, match="must be fitted"):
            get_model(name).predict(3)

    def test_predict_returns_exactly_the_horizon(self, name, daily_series):
        model = get_model(name).fit(daily_series)
        for horizon in (1, 7, 21):
            assert model.predict(horizon).shape == (horizon,)

    @pytest.mark.parametrize("horizon", [0, -3])
    def test_rejects_non_positive_horizon(self, name, daily_series, horizon):
        model = get_model(name).fit(daily_series)
        with pytest.raises(ValueError, match="horizon must be positive"):
            model.predict(horizon)

    def test_rejects_unsorted_history(self, name, daily_series):
        shuffled = daily_series.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="sorted by date"):
            get_model(name).fit(shuffled)

    def test_rejects_nan_targets(self, name, daily_series):
        broken = daily_series.copy()
        broken.loc[5, "units"] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            get_model(name).fit(broken)

    def test_rejects_empty_history(self, name, daily_series):
        with pytest.raises(ValueError, match="empty"):
            get_model(name).fit(daily_series.iloc[0:0])

    def test_refitting_on_more_data_does_not_carry_state(self, name, daily_series):
        model = get_model(name)
        model.fit(daily_series.iloc[:100])
        first = model.predict(7)
        model.fit(daily_series.iloc[:150])
        second = model.predict(7)

        fresh = get_model(name).fit(daily_series.iloc[:150]).predict(7)
        np.testing.assert_allclose(second, fresh)
        assert not np.allclose(first, second)

    def test_is_a_forecaster(self, name):
        assert isinstance(get_model(name), Forecaster)


class TestRegistry:
    def test_lists_the_built_in_models(self):
        assert {"seasonal_naive", "mean", "sarimax"} <= set(available_models())

    def test_baseline_is_registered(self):
        assert BASELINE_MODEL in available_models()

    def test_returns_a_new_instance_each_call(self):
        assert get_model("mean") is not get_model("mean")

    def test_passes_params_through(self):
        model = get_model("seasonal_naive", season_length=14)
        assert model.season_length == 14

    def test_unknown_model_lists_alternatives(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("prophet")

    def test_registering_a_duplicate_needs_overwrite(self):
        with pytest.raises(ValueError, match="already registered"):
            register_model("mean", MeanForecaster)

    def test_registering_a_new_model_makes_it_available(self):
        register_model("mean_copy", MeanForecaster)
        try:
            assert "mean_copy" in available_models()
            assert isinstance(get_model("mean_copy"), MeanForecaster)
        finally:
            from forecasting import registry

            registry._REGISTRY.pop("mean_copy", None)

    def test_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_model("", MeanForecaster)
