"""Tests for the eval gate and data generation scripts."""

from __future__ import annotations

import json

import pytest
from check_eval_gate import evaluate, main, render_summary
from generate_data import generate


class TestEvaluateChecks:
    def test_clean_metrics_pass_every_check(self, metrics_payload, thresholds):
        checks = evaluate(metrics_payload, thresholds)
        assert checks
        assert all(check.passed for check in checks)

    def test_mape_above_the_ceiling_fails(self, metrics_payload, thresholds):
        metrics_payload["models"]["sarimax"]["metrics"]["mape"] = 0.25
        failures = [c for c in evaluate(metrics_payload, thresholds) if not c.passed]
        assert any("mape" in check.name for check in failures)

    def test_mae_above_the_ceiling_fails(self, metrics_payload, thresholds):
        metrics_payload["models"]["sarimax"]["metrics"]["mae"] = 500.0
        failures = [c for c in evaluate(metrics_payload, thresholds) if not c.passed]
        assert any("mae" in check.name for check in failures)

    def test_large_negative_bias_fails(self, metrics_payload, thresholds):
        metrics_payload["models"]["sarimax"]["metrics"]["bias"] = -200.0
        failures = [c for c in evaluate(metrics_payload, thresholds) if not c.passed]
        assert any("bias" in check.name for check in failures)

    def test_champion_that_ties_the_baseline_fails_the_improvement_check(
        self, metrics_payload, thresholds
    ):
        metrics_payload["models"]["sarimax"]["metrics"]["mape"] = 0.10
        failures = [c for c in evaluate(metrics_payload, thresholds) if not c.passed]
        assert any("beats" in check.name for check in failures)

    def test_champion_worse_than_baseline_fails(self, metrics_payload, thresholds):
        metrics_payload["models"]["sarimax"]["metrics"]["mape"] = 0.12
        failures = [c for c in evaluate(metrics_payload, thresholds) if not c.passed]
        assert any("beats" in check.name for check in failures)

    def test_too_few_origins_fails(self, metrics_payload, thresholds):
        metrics_payload["models"]["sarimax"]["n_splits"] = 2
        failures = [c for c in evaluate(metrics_payload, thresholds) if not c.passed]
        assert any("origins" in check.name for check in failures)

    def test_baseline_comparison_can_be_disabled(self, metrics_payload, thresholds):
        thresholds["must_beat_baseline"] = False
        metrics_payload["models"]["sarimax"]["metrics"]["mape"] = 0.10
        assert all(check.passed for check in evaluate(metrics_payload, thresholds))

    def test_absent_thresholds_are_skipped(self, metrics_payload):
        checks = evaluate(metrics_payload, {"max_mape": 0.11})
        assert len(checks) == 1

    def test_missing_champion_raises(self, metrics_payload, thresholds):
        del metrics_payload["models"]["sarimax"]
        with pytest.raises(KeyError, match="Champion model"):
            evaluate(metrics_payload, thresholds)

    def test_missing_baseline_raises(self, metrics_payload, thresholds):
        del metrics_payload["models"]["seasonal_naive"]
        with pytest.raises(KeyError, match="Baseline model"):
            evaluate(metrics_payload, thresholds)


class TestRenderSummary:
    def test_passing_summary_says_passed(self, metrics_payload, thresholds):
        summary = render_summary(metrics_payload, evaluate(metrics_payload, thresholds))
        assert "passed" in summary
        assert "FAIL" not in summary

    def test_failing_summary_flags_the_check(self, metrics_payload, thresholds):
        metrics_payload["models"]["sarimax"]["metrics"]["mape"] = 0.5
        summary = render_summary(metrics_payload, evaluate(metrics_payload, thresholds))
        assert "failed" in summary
        assert "**FAIL**" in summary

    def test_every_model_appears_in_the_table(self, metrics_payload, thresholds):
        summary = render_summary(metrics_payload, evaluate(metrics_payload, thresholds))
        for name in metrics_payload["models"]:
            assert name in summary

    def test_every_markdown_table_has_consistent_columns(self, metrics_payload, thresholds):
        summary = render_summary(metrics_payload, evaluate(metrics_payload, thresholds))
        rows = [line for line in summary.splitlines() if line.startswith("|")]

        # A 5-column metrics table and a 4-column checks table, each with a
        # header, a separator, and one row per entry.
        metrics_rows = [line for line in rows if line.count("|") == 6]
        check_rows = [line for line in rows if line.count("|") == 5]

        assert len(metrics_rows) == 2 + len(metrics_payload["models"])
        assert len(check_rows) == 2 + len(evaluate(metrics_payload, thresholds))
        assert len(metrics_rows) + len(check_rows) == len(rows)


class TestGateCli:
    def _write(self, tmp_path, metrics_payload, thresholds):
        metrics_file = tmp_path / "metrics.json"
        thresholds_file = tmp_path / "thresholds.yml"
        summary_file = tmp_path / "summary.md"
        metrics_file.write_text(json.dumps(metrics_payload), encoding="utf-8")
        thresholds_file.write_text(json.dumps(thresholds), encoding="utf-8")
        return metrics_file, thresholds_file, summary_file

    def test_exits_zero_when_everything_passes(
        self, tmp_path, metrics_payload, thresholds, monkeypatch
    ):
        metrics_file, thresholds_file, summary_file = self._write(
            tmp_path, metrics_payload, thresholds
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_eval_gate.py",
                "--metrics",
                str(metrics_file),
                "--thresholds",
                str(thresholds_file),
                "--summary",
                str(summary_file),
            ],
        )
        assert main() == 0
        assert summary_file.exists()

    def test_exits_one_on_regression(self, tmp_path, metrics_payload, thresholds, monkeypatch):
        metrics_payload["models"]["sarimax"]["metrics"]["mape"] = 0.9
        metrics_file, thresholds_file, summary_file = self._write(
            tmp_path, metrics_payload, thresholds
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_eval_gate.py",
                "--metrics",
                str(metrics_file),
                "--thresholds",
                str(thresholds_file),
                "--summary",
                str(summary_file),
            ],
        )
        assert main() == 1
        assert "**FAIL**" in summary_file.read_text(encoding="utf-8")

    def test_warn_only_downgrades_a_regression(
        self, tmp_path, metrics_payload, thresholds, monkeypatch
    ):
        metrics_payload["models"]["sarimax"]["metrics"]["mape"] = 0.9
        metrics_file, thresholds_file, summary_file = self._write(
            tmp_path, metrics_payload, thresholds
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_eval_gate.py",
                "--metrics",
                str(metrics_file),
                "--thresholds",
                str(thresholds_file),
                "--summary",
                str(summary_file),
                "--warn-only",
            ],
        )
        assert main() == 0

    def test_missing_metrics_file_exits_two(self, tmp_path, thresholds, monkeypatch):
        thresholds_file = tmp_path / "thresholds.yml"
        thresholds_file.write_text(json.dumps(thresholds), encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_eval_gate.py",
                "--metrics",
                str(tmp_path / "absent.json"),
                "--thresholds",
                str(thresholds_file),
                "--summary",
                str(tmp_path / "summary.md"),
            ],
        )
        assert main() == 2


class TestGenerateData:
    def test_is_deterministic_for_a_given_seed(self):
        import pandas as pd

        pd.testing.assert_frame_equal(generate(seed=7), generate(seed=7))

    def test_different_seeds_produce_different_data(self):
        assert not generate(seed=7)["units"].equals(generate(seed=8)["units"])

    def test_shape_and_columns(self):
        frame = generate(seed=7)
        assert list(frame.columns) == ["date", "sku", "units", "price", "on_promotion"]
        assert frame["sku"].nunique() == 3
        assert len(frame) == 3 * 1096

    def test_units_are_non_negative(self):
        assert (generate(seed=7)["units"] >= 0).all()

    def test_no_gaps_in_the_daily_index(self):
        for _, group in generate(seed=7).groupby("sku"):
            dates = group["date"].sort_values()
            assert (dates.diff().dropna().dt.days == 1).all()

    def test_promotions_occur_but_are_not_constant(self):
        flags = generate(seed=7)["on_promotion"]
        assert 0 < flags.mean() < 0.5

    def test_promotional_days_are_discounted(self):
        frame = generate(seed=7)
        for _, group in frame.groupby("sku"):
            promo_price = group.loc[group["on_promotion"], "price"].max()
            list_price = group.loc[~group["on_promotion"], "price"].min()
            assert promo_price < list_price

    def test_committed_csv_matches_the_generator(self, project_root):
        """Guards against someone hand-editing data/raw/sales.csv."""
        import pandas as pd

        committed = pd.read_csv(project_root / "data" / "raw" / "sales.csv", parse_dates=["date"])
        expected = generate(seed=7)
        pd.testing.assert_frame_equal(
            committed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )
