#!/usr/bin/env python3
"""Compare reports/metrics.json against eval/thresholds.yml.

Exits 1 when the champion model regresses, and always writes a markdown summary
that CI posts as a pull request comment.

Usage:
    python scripts/check_eval_gate.py
    python scripts/check_eval_gate.py --metrics reports/metrics.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = PROJECT_ROOT / "reports" / "metrics.json"
DEFAULT_THRESHOLDS = PROJECT_ROOT / "eval" / "thresholds.yml"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "eval_summary.md"

PASS = "pass"
FAIL = "fail"


@dataclass
class Check:
    name: str
    actual: float
    bound: float
    passed: bool
    detail: str

    @property
    def status(self) -> str:
        return PASS if self.passed else FAIL


def evaluate(metrics: dict, thresholds: dict) -> list[Check]:
    """Run every configured check and return the results in report order."""
    config = metrics["config"]
    models = metrics["models"]

    champion_name = config["champion_model"]
    baseline_name = config["baseline_model"]

    if champion_name not in models:
        raise KeyError(f"Champion model {champion_name!r} is missing from the metrics file")

    champion = models[champion_name]["metrics"]
    checks: list[Check] = []

    for metric, key in (("mape", "max_mape"), ("mae", "max_mae"), ("rmse", "max_rmse")):
        bound = thresholds.get(key)
        if bound is None:
            continue
        actual = float(champion[metric])
        checks.append(
            Check(
                name=f"{champion_name} {metric} <= {bound}",
                actual=actual,
                bound=float(bound),
                passed=actual <= float(bound),
                detail=f"{actual:.4f}",
            )
        )

    max_abs_bias = thresholds.get("max_abs_bias")
    if max_abs_bias is not None:
        actual = abs(float(champion["bias"]))
        checks.append(
            Check(
                name=f"{champion_name} abs(bias) <= {max_abs_bias}",
                actual=actual,
                bound=float(max_abs_bias),
                passed=actual <= float(max_abs_bias),
                detail=f"{actual:.2f}",
            )
        )

    min_splits = thresholds.get("min_splits")
    if min_splits is not None:
        actual = float(models[champion_name]["n_splits"])
        checks.append(
            Check(
                name=f"backtest origins >= {min_splits}",
                actual=actual,
                bound=float(min_splits),
                passed=actual >= float(min_splits),
                detail=f"{int(actual)}",
            )
        )

    if thresholds.get("must_beat_baseline"):
        if baseline_name not in models:
            raise KeyError(f"Baseline model {baseline_name!r} is missing from the metrics file")
        baseline_mape = float(models[baseline_name]["metrics"]["mape"])
        champion_mape = float(champion["mape"])
        improvement = (baseline_mape - champion_mape) / baseline_mape if baseline_mape > 0 else 0.0
        minimum = float(thresholds.get("min_relative_improvement", 0.0))
        checks.append(
            Check(
                name=f"{champion_name} beats {baseline_name} by >= {minimum:.0%} MAPE",
                actual=improvement,
                bound=minimum,
                passed=improvement >= minimum,
                detail=f"{improvement:+.2%}",
            )
        )

    return checks


def render_summary(metrics: dict, checks: list[Check]) -> str:
    """Build the markdown block CI posts on the pull request."""
    config = metrics["config"]
    models = metrics["models"]
    champion_name = config["champion_model"]
    failed = [check for check in checks if not check.passed]

    headline = (
        f"Model eval gate **failed** ({len(failed)} of {len(checks)} checks)"
        if failed
        else f"Model eval gate **passed** ({len(checks)} checks)"
    )

    lines = [
        "## Model eval gate",
        "",
        headline,
        "",
        f"Series `{config['series']}`, horizon {config['horizon']} days, "
        f"{config['n_splits']} rolling origins. Champion: `{champion_name}`.",
        "",
        "| model | MAPE | MAE | RMSE | bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for name, result in models.items():
        values = result["metrics"]
        label = f"**{name}**" if name == champion_name else name
        lines.append(
            f"| {label} | {values['mape']:.2%} | {values['mae']:.2f} | "
            f"{values['rmse']:.2f} | {values['bias']:+.2f} |"
        )

    lines += ["", "| check | value | threshold | result |", "| --- | ---: | ---: | --- |"]
    for check in checks:
        marker = "pass" if check.passed else "**FAIL**"
        lines.append(f"| {check.name} | {check.detail} | {check.bound:g} | {marker} |")

    if failed:
        lines += [
            "",
            "Raising a threshold in `eval/thresholds.yml` to clear this gate needs an "
            "explicit reason in the pull request description.",
        ]

    lines += ["", f"<sub>Generated {metrics['generated_at']} from `reports/metrics.json`.</sub>"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report failures but exit 0 (used for exploratory runs, never in CI)",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"error: {args.metrics} not found. Run `make backtest` first.")
        return 2

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8")) or {}

    checks = evaluate(metrics, thresholds)
    summary = render_summary(metrics, checks)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary, encoding="utf-8")
    print(summary)

    failed = [check for check in checks if not check.passed]
    if failed and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
