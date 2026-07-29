"""Loading and splitting behaviour, especially the chronological guarantees."""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from forecasting.data import (
    aggregate_total,
    load_sales,
    rolling_origin_splits,
    select_sku,
    train_test_split,
)


class TestLoadSales:
    def test_loads_the_committed_dataset(self):
        frame = load_sales()
        assert not frame.empty
        assert {"date", "sku", "units", "price", "on_promotion"} <= set(frame.columns)
        assert pd.api.types.is_datetime64_any_dtype(frame["date"])
        assert frame["units"].dtype == float

    def test_one_row_per_sku_and_date(self):
        frame = load_sales()
        assert not frame.duplicated(subset=["sku", "date"]).any()

    def test_sorted_by_sku_then_date(self):
        frame = load_sales()
        for _, group in frame.groupby("sku"):
            assert group["date"].is_monotonic_increasing

    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make data"):
            load_sales(tmp_path / "nope.csv")

    def test_missing_columns_are_reported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("date,value\n2023-01-01,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing columns"):
            load_sales(path)


class TestSelectAndAggregate:
    def test_select_sku_returns_one_series(self, multi_sku_frame):
        subset = select_sku(multi_sku_frame, "SKU-A")
        assert subset["sku"].unique().tolist() == ["SKU-A"]
        assert len(subset) == 60

    def test_select_unknown_sku_lists_options(self, multi_sku_frame):
        with pytest.raises(KeyError, match="SKU-A"):
            select_sku(multi_sku_frame, "SKU-NOPE")

    def test_aggregate_total_sums_across_skus(self, multi_sku_frame):
        total = aggregate_total(multi_sku_frame)
        assert len(total) == 60
        assert total["units"].sum() == multi_sku_frame["units"].sum()
        assert total["sku"].unique().tolist() == ["TOTAL"]


class TestTrainTestSplit:
    def test_test_set_is_the_final_horizon(self, daily_series):
        split = train_test_split(daily_series, horizon=14)
        assert split.horizon == 14
        assert len(split.train) == len(daily_series) - 14
        assert split.test["date"].min() > split.train["date"].max()

    def test_cutoff_is_the_last_training_date(self, daily_series):
        split = train_test_split(daily_series, horizon=14)
        assert split.cutoff == split.train["date"].max()

    @pytest.mark.parametrize("horizon", [0, -1])
    def test_rejects_non_positive_horizon(self, daily_series, horizon):
        with pytest.raises(ValueError, match="horizon must be positive"):
            train_test_split(daily_series, horizon=horizon)

    def test_rejects_horizon_that_consumes_all_data(self, daily_series):
        with pytest.raises(ValueError, match="no training data"):
            train_test_split(daily_series, horizon=len(daily_series))

    def test_rejects_multi_sku_frames(self, multi_sku_frame):
        with pytest.raises(ValueError, match="single series"):
            train_test_split(multi_sku_frame, horizon=5)

    def test_rejects_duplicate_dates(self, daily_series):
        doubled = pd.concat([daily_series, daily_series.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate dates"):
            train_test_split(doubled, horizon=5)


class TestRollingOriginSplits:
    def test_yields_the_requested_number_of_splits(self, daily_series):
        splits = list(rolling_origin_splits(daily_series, horizon=14, n_splits=5))
        assert len(splits) == 5

    def test_training_windows_expand(self, daily_series):
        splits = list(rolling_origin_splits(daily_series, horizon=14, n_splits=5))
        sizes = [len(split.train) for split in splits]
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)

    def test_no_test_row_predates_its_own_training_window(self, daily_series):
        for split in rolling_origin_splits(daily_series, horizon=14, n_splits=5):
            assert split.test["date"].min() > split.train["date"].max()

    def test_final_split_ends_at_the_series_end(self, daily_series):
        splits = list(rolling_origin_splits(daily_series, horizon=14, n_splits=5))
        assert splits[-1].test["date"].max() == daily_series["date"].max()

    def test_step_controls_the_gap_between_origins(self, daily_series):
        splits = list(rolling_origin_splits(daily_series, horizon=14, n_splits=4, step=7))
        sizes = [len(split.train) for split in splits]
        assert [b - a for a, b in pairwise(sizes)] == [7, 7, 7]

    def test_every_test_window_has_the_full_horizon(self, daily_series):
        for split in rolling_origin_splits(daily_series, horizon=14, n_splits=5):
            assert split.horizon == 14

    def test_series_too_short_raises(self, daily_series):
        with pytest.raises(ValueError, match="too short"):
            list(rolling_origin_splits(daily_series, horizon=60, n_splits=10))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"horizon": 0, "n_splits": 3}, "horizon must be positive"),
            ({"horizon": 5, "n_splits": 0}, "n_splits must be positive"),
            ({"horizon": 5, "n_splits": 3, "step": 0}, "step must be positive"),
        ],
    )
    def test_rejects_invalid_arguments(self, daily_series, kwargs, message):
        with pytest.raises(ValueError, match=message):
            list(rolling_origin_splits(daily_series, **kwargs))
