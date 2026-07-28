#!/usr/bin/env python3
"""Run the rolling-origin backtest and write reports/metrics.json.

This is the artifact the eval gate reads, so it is deliberately boring: fixed
config, deterministic data, one JSON file out.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --horizon 7 --n-splits 8 --plot
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from forecasting.data import aggregate_total, load_sales, select_sku
from forecasting.evaluate import backtest
from forecasting.registry import BASELINE_MODEL, available_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "metrics.json"

CHAMPION_MODEL = "sarimax"
DEFAULT_MODELS = (BASELINE_MODEL, "mean", CHAMPION_MODEL)
MODEL_PARAMS: dict[str, dict[str, object]] = {
    "seasonal_naive": {"season_length": 7},
    "mean": {"window": 28},
    "sarimax": {},
}


def run(
    horizon: int = 14,
    n_splits: int = 5,
    models: tuple[str, ...] = DEFAULT_MODELS,
    series: str = "TOTAL",
    data_path: Path | None = None,
) -> dict[str, object]:
    """Backtest every model on one series and return the metrics payload."""
    frame = load_sales(data_path)
    target = aggregate_total(frame) if series == "TOTAL" else select_sku(frame, series)

    results: dict[str, object] = {}
    for name in models:
        if name not in available_models():
            raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
        result = backtest(
            target,
            model=name,
            horizon=horizon,
            n_splits=n_splits,
            model_params=MODEL_PARAMS.get(name, {}),
        )
        results[name] = result.to_dict()

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "rows": len(frame),
            "skus": sorted(frame["sku"].unique().tolist()),
            "start": str(pd.Timestamp(frame["date"].min()).date()),
            "end": str(pd.Timestamp(frame["date"].max()).date()),
        },
        "config": {
            "series": series,
            "horizon": horizon,
            "n_splits": n_splits,
            "baseline_model": BASELINE_MODEL,
            "champion_model": CHAMPION_MODEL if CHAMPION_MODEL in results else models[-1],
        },
        "models": results,
    }


def write_plot(payload_models: dict[str, object], output: Path) -> None:
    """Save a bar chart of MAPE by model next to the metrics file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(payload_models)
    values = [payload_models[name]["metrics"]["mape"] * 100 for name in names]

    figure, axes = plt.subplots(figsize=(6, 3.5))
    axes.bar(names, values, color="#4c72b0")
    axes.set_ylabel("MAPE (%)")
    axes.set_title("Backtest accuracy by model (lower is better)")
    for index, value in enumerate(values):
        axes.text(index, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon in days")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of rolling origins")
    parser.add_argument("--series", default="TOTAL", help="'TOTAL' or a specific SKU")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--data", type=Path, default=None, help="Override the sales CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true", help="Also write reports/mape_by_model.png")
    args = parser.parse_args()

    payload = run(
        horizon=args.horizon,
        n_splits=args.n_splits,
        models=tuple(args.models),
        series=args.series,
        data_path=args.data,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")

    for name, result in payload["models"].items():
        metrics = result["metrics"]
        print(f"  {name:<16} mape={metrics['mape']:.4f}  mae={metrics['mae']:.2f}")

    if args.plot:
        plot_path = args.output.with_name("mape_by_model.png")
        write_plot(payload["models"], plot_path)
        print(f"Wrote {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
