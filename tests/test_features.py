"""Feature engineering tests.

The important ones are the look-ahead guards. If a change makes a feature at
time t depend on the target at time t, these fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.features import (
    add_calendar_features,
    add_fourier_terms,
    add_lag_features,
    add_promotion_features,
    add_rolling_features,
    build_feature_frame,
    feature_columns,
)


class TestCalendarFeatures:
    def test_adds_expected_columns(self, daily_series):
        out = add_calendar_features(daily_series)
        expected = {
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        }
        assert expected <= set(out.columns)

    def test_values_match_the_timestamp(self):
        frame = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01", "2023-06-30"]), "units": [1.0, 2.0]}
        )
        out = add_calendar_features(frame)
        assert out.loc[0, "day_of_week"] == 6  # Sunday
        assert bool(out.loc[0, "is_weekend"]) is True
        assert bool(out.loc[0, "is_month_start"]) is True
        assert out.loc[1, "month"] == 6
        assert bool(out.loc[1, "is_month_end"]) is True

    def test_does_not_depend_on_the_target(self, daily_series):
        """Calendar features are timestamp-only, so scrambling the target changes nothing."""
        scrambled = daily_series.copy()
        scrambled["units"] = scrambled["units"].to_numpy()[::-1]

        original = add_calendar_features(daily_series).drop(columns=["units"])
        shuffled = add_calendar_features(scrambled).drop(columns=["units"])
        pd.testing.assert_frame_equal(original, shuffled)

    def test_original_frame_is_not_mutated(self, daily_series):
        before = list(daily_series.columns)
        add_calendar_features(daily_series)
        assert list(daily_series.columns) == before


class TestFourierTerms:
    def test_adds_a_sin_cos_pair_per_order(self, daily_series):
        out = add_fourier_terms(daily_series, order=3)
        for k in (1, 2, 3):
            assert f"yearly_sin_{k}" in out.columns
            assert f"yearly_cos_{k}" in out.columns

    def test_terms_are_bounded(self, daily_series):
        out = add_fourier_terms(daily_series, order=2)
        for column in [c for c in out.columns if c.startswith("yearly_")]:
            assert out[column].abs().max() <= 1.0 + 1e-12

    def test_weekly_period_repeats_every_seven_days(self, daily_series):
        out = add_fourier_terms(daily_series, period=7.0, order=1, prefix="weekly")
        assert out.loc[0, "weekly_sin_1"] == pytest.approx(out.loc[7, "weekly_sin_1"], abs=1e-9)

    def test_rejects_order_below_one(self, daily_series):
        with pytest.raises(ValueError, match="order must be at least 1"):
            add_fourier_terms(daily_series, order=0)

    def test_phase_is_anchored_to_a_fixed_epoch_not_the_frame(self, daily_series):
        """Regression: a calendar date must encode identically in every fold.

        Anchoring the phase to ``dates.min()`` made the encoding depend on which
        slice was passed in, so a coefficient learned on one backtest fold meant
        something different on the next.
        """
        early = daily_series.iloc[:120].reset_index(drop=True)
        late = daily_series.iloc[60:].reset_index(drop=True)

        early_out = add_fourier_terms(early, order=2).set_index("date")
        late_out = add_fourier_terms(late, order=2).set_index("date")

        shared = early_out.index.intersection(late_out.index)
        assert len(shared) == 60

        for column in [c for c in early_out.columns if c.startswith("yearly_")]:
            pd.testing.assert_series_equal(
                early_out.loc[shared, column],
                late_out.loc[shared, column],
            )

    def test_encoding_depends_only_on_the_date(self, daily_series):
        """The same day encodes the same way even in a one-row frame."""
        single = daily_series.iloc[[100]].reset_index(drop=True)

        full_out = add_fourier_terms(daily_series, order=1)
        single_out = add_fourier_terms(single, order=1)

        assert single_out.loc[0, "yearly_sin_1"] == pytest.approx(full_out.loc[100, "yearly_sin_1"])
        assert single_out.loc[0, "yearly_cos_1"] == pytest.approx(full_out.loc[100, "yearly_cos_1"])


class TestLagFeatures:
    def test_lag_holds_the_earlier_value(self, daily_series):
        out = add_lag_features(daily_series, lags=(1, 7))
        assert out.loc[10, "lag_1"] == pytest.approx(daily_series.loc[9, "units"])
        assert out.loc[10, "lag_7"] == pytest.approx(daily_series.loc[3, "units"])

    def test_leading_rows_are_missing(self, daily_series):
        out = add_lag_features(daily_series, lags=(7,))
        assert out["lag_7"].isna().sum() == 7

    def test_never_equals_the_current_target_by_construction(self, daily_series):
        """A lag column must never be a copy of the unshifted target."""
        out = add_lag_features(daily_series, lags=(1, 7, 14))
        for lag in (1, 7, 14):
            aligned = out[[f"lag_{lag}", "units"]].dropna()
            assert not np.allclose(aligned[f"lag_{lag}"], aligned["units"])

    def test_rejects_zero_and_negative_lags(self, daily_series):
        for lags in ((0,), (-1,), (1, 0)):
            with pytest.raises(ValueError, match="lags must be >= 1"):
                add_lag_features(daily_series, lags=lags)


class TestRollingFeatures:
    def test_window_excludes_the_current_observation(self, daily_series):
        out = add_rolling_features(daily_series, windows=(3,), min_lag=1)
        # roll_mean_3 at row 10 averages rows 7, 8, 9 and must not touch row 10.
        expected = daily_series.loc[7:9, "units"].mean()
        assert out.loc[10, "roll_mean_3"] == pytest.approx(expected)

    def test_changing_only_the_last_target_leaves_features_untouched(self, daily_series):
        """The strongest leakage check: perturb the future, nothing earlier moves."""
        perturbed = daily_series.copy()
        perturbed.loc[perturbed.index[-1], "units"] += 10_000.0

        original = add_rolling_features(daily_series, windows=(7,))
        modified = add_rolling_features(perturbed, windows=(7,))

        pd.testing.assert_series_equal(
            original["roll_mean_7"].iloc[:-1], modified["roll_mean_7"].iloc[:-1]
        )

    def test_rejects_min_lag_below_one(self, daily_series):
        with pytest.raises(ValueError, match="min_lag must be >= 1"):
            add_rolling_features(daily_series, min_lag=0)

    def test_rejects_windows_below_two(self, daily_series):
        with pytest.raises(ValueError, match="windows must be >= 2"):
            add_rolling_features(daily_series, windows=(1,))


class TestPromotionFeatures:
    def test_adds_expected_columns(self, promotion_series):
        out = add_promotion_features(promotion_series)
        assert "on_promotion" in out.columns
        assert "days_until_next_promotion" in out.columns

    def test_on_promotion_is_numeric(self, promotion_series):
        out = add_promotion_features(promotion_series)
        assert pd.api.types.is_numeric_dtype(out["on_promotion"])

    def test_on_promotion_day_days_until_is_zero(self, promotion_series):
        out = add_promotion_features(promotion_series)
        promo_rows = out[out["on_promotion"] == 1]
        assert (promo_rows["days_until_next_promotion"] == 0).all()

    def test_days_until_counts_correctly(self, promotion_series):
        """Row immediately before the first promo window should be 1 day away."""
        out = add_promotion_features(promotion_series)
        # Index 19 is the day before the first promotion window (index 20).
        assert out.loc[19, "days_until_next_promotion"] == 1.0

    def test_days_until_nan_after_last_promotion(self, promotion_series):
        """Rows after the last promotion date in the frame have no next promo."""
        # Remove the final promo day (index 99) so the last promo ends at index 65.
        series = promotion_series.copy()
        series.loc[99, "on_promotion"] = False
        out = add_promotion_features(series)
        # Rows after index 65 should have NaN.
        assert out.loc[66:, "days_until_next_promotion"].isna().all()

    def test_perturbing_last_target_does_not_change_earlier_promotion_features(
        self, promotion_series
    ):
        """Promotion features must not depend on the target value.

        Changing ``units`` at the last row (a promotion day) should leave every
        earlier row's promotion features identical.  This is the look-ahead guard
        for target-independent exogenous features.
        """
        perturbed = promotion_series.copy()
        perturbed.loc[perturbed.index[-1], "units"] += 10_000.0

        original = add_promotion_features(promotion_series)
        modified = add_promotion_features(perturbed)

        pd.testing.assert_series_equal(
            original["on_promotion"].iloc[:-1],
            modified["on_promotion"].iloc[:-1],
        )
        pd.testing.assert_series_equal(
            original["days_until_next_promotion"].iloc[:-1],
            modified["days_until_next_promotion"].iloc[:-1],
        )

    def test_days_until_does_not_require_units_at_future_promotion_date(self, promotion_series):
        """days_until_next_promotion is derived from the promotion schedule alone.

        Setting future ``units`` to NaN (unknown at forecast time) must not
        prevent computation, proving the column is a genuinely exogenous input.
        """
        unknown_future = promotion_series.copy()
        # Blank out all units from the second promo window onward.
        unknown_future.loc[60:, "units"] = np.nan

        out = add_promotion_features(unknown_future)
        # Rows before the second promo window should still count down correctly.
        assert out.loc[59, "days_until_next_promotion"] == 1.0
        assert out.loc[55, "days_until_next_promotion"] == 5.0

    def test_rejects_frame_without_on_promotion(self, daily_series):
        with pytest.raises(ValueError, match="on_promotion"):
            add_promotion_features(daily_series)

    def test_original_frame_is_not_mutated(self, promotion_series):
        before = list(promotion_series.columns)
        add_promotion_features(promotion_series)
        assert list(promotion_series.columns) == before


class TestBuildFeatureFrame:
    def test_drops_the_warmup_rows(self, daily_series):
        out = build_feature_frame(daily_series, lags=(1, 7, 28), rolling_windows=(28,))
        assert out.notna().all().all()
        assert len(out) < len(daily_series)

    def test_keeps_the_target_and_date(self, daily_series):
        out = build_feature_frame(daily_series)
        assert "units" in out.columns
        assert "date" in out.columns

    def test_output_stays_chronological(self, daily_series):
        out = build_feature_frame(daily_series)
        assert out["date"].is_monotonic_increasing

    def test_dropna_false_preserves_row_count(self, daily_series):
        out = build_feature_frame(daily_series, dropna=False)
        assert len(out) == len(daily_series)

    def test_no_feature_correlates_perfectly_with_the_target(self):
        """A perfect correlation is the signature of an accidentally leaked target.

        Run against the real dataset rather than the smooth fixture: a noiseless
        sine wave makes ``lag_7`` an exact linear function of the target, which
        would trip this check for reasons that have nothing to do with leakage.
        """
        from forecasting.data import aggregate_total, load_sales

        out = build_feature_frame(aggregate_total(load_sales()))
        for column in feature_columns(out):
            if out[column].nunique() <= 1:
                continue  # constant over this window, correlation is undefined
            correlation = out[column].corr(out["units"])
            assert abs(correlation) < 0.99, f"{column} looks like a leaked copy of the target"


class TestFeatureColumns:
    def test_excludes_identifiers_and_the_target(self, daily_series):
        out = build_feature_frame(daily_series)
        columns = feature_columns(out)
        assert "units" not in columns
        assert "date" not in columns
        assert "sku" not in columns
        assert columns


class TestMultiSeriesGuard:
    """A bare shift over interleaved SKUs is cross-series leakage, not a lag.

    ``build_feature_frame`` sorts by date only, so on a multi-SKU frame
    ``lag_1`` for one SKU would hold another SKU's value for the *same* date.
    The shape and dtypes come out fine, which is exactly why it has to raise.
    """

    def test_build_feature_frame_rejects_multiple_skus(self, multi_sku_frame):
        with pytest.raises(ValueError, match="single series"):
            build_feature_frame(multi_sku_frame)

    def test_add_lag_features_rejects_multiple_skus(self, multi_sku_frame):
        with pytest.raises(ValueError, match="single series"):
            add_lag_features(multi_sku_frame)

    def test_add_rolling_features_rejects_multiple_skus(self, multi_sku_frame):
        with pytest.raises(ValueError, match="single series"):
            add_rolling_features(multi_sku_frame)

    def test_the_error_names_the_way_out(self, multi_sku_frame):
        with pytest.raises(ValueError) as info:
            build_feature_frame(multi_sku_frame)
        assert "select_sku" in str(info.value)
        assert "aggregate_total" in str(info.value)

    def test_aggregating_first_works(self, multi_sku_frame):
        from forecasting.data import aggregate_total

        out = build_feature_frame(aggregate_total(multi_sku_frame))
        assert not out.empty

    def test_selecting_one_sku_works(self, multi_sku_frame):
        from forecasting.data import select_sku

        out = build_feature_frame(select_sku(multi_sku_frame, "SKU-A"), dropna=False)
        assert len(out) == 60

    def test_duplicate_dates_within_one_sku_are_rejected(self, daily_series):
        doubled = pd.concat([daily_series, daily_series], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate dates"):
            build_feature_frame(doubled)
