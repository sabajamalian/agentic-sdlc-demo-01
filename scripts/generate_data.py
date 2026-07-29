#!/usr/bin/env python3
"""Generate the synthetic daily sales dataset.

Deterministic given the seed, so CI, notebooks and every agent-authored change
see byte-identical data. The generated CSV is committed to the repo, which keeps
notebooks runnable with no build step and keeps CI hermetic.

Usage:
    python scripts/generate_data.py [--seed 7] [--output data/raw/sales.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "sales.csv"

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

# base level, weekly amplitude, yearly amplitude, trend per year, promo lift, list price
SKUS: dict[str, dict[str, float]] = {
    "SKU-ALPHA": {
        "base": 220.0,
        "weekly": 0.28,
        "yearly": 0.18,
        "trend": 0.12,
        "promo_lift": 0.45,
        "price": 24.99,
        "noise": 0.09,
    },
    "SKU-BRAVO": {
        "base": 95.0,
        "weekly": 0.42,
        "yearly": 0.30,
        "trend": -0.05,
        "promo_lift": 0.70,
        "price": 12.50,
        "noise": 0.14,
    },
    "SKU-CHARLIE": {
        "base": 410.0,
        "weekly": 0.15,
        "yearly": 0.08,
        "trend": 0.04,
        "promo_lift": 0.25,
        "price": 8.75,
        "noise": 0.07,
    },
}

# Weekday multipliers, Monday first. Retail-ish: quiet midweek, busy weekend.
WEEKDAY_SHAPE = np.array([0.92, 0.88, 0.90, 0.98, 1.15, 1.28, 0.89])


def generate(seed: int = 7) -> pd.DataFrame:
    """Build the full multi-SKU daily sales frame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    day_index = np.arange(len(dates), dtype=float)
    years_elapsed = day_index / 365.25

    frames = []
    for sku, cfg in SKUS.items():
        weekday = WEEKDAY_SHAPE[dates.dayofweek.to_numpy()] ** cfg["weekly"]
        day_of_year = dates.dayofyear.to_numpy()
        yearly = 1.0 + cfg["yearly"] * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        trend = 1.0 + cfg["trend"] * years_elapsed

        # Promotions run in blocks, so their effect is autocorrelated the way a
        # real promo calendar is rather than sprinkled at random.
        on_promotion = _promotion_blocks(rng, len(dates))
        promo = 1.0 + cfg["promo_lift"] * on_promotion

        # Late-December lift for the two consumer SKUs.
        holiday = np.where(
            (dates.month == 12) & (dates.day >= 15),
            1.0 + (0.35 if sku != "SKU-CHARLIE" else 0.10),
            1.0,
        )

        noise = rng.normal(1.0, cfg["noise"], size=len(dates))
        units = cfg["base"] * weekday * yearly * trend * promo * holiday * noise
        units = np.clip(np.round(units), 0, None)

        price = np.where(on_promotion, round(cfg["price"] * 0.85, 2), cfg["price"])

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku": sku,
                    "units": units,
                    "price": price,
                    "on_promotion": on_promotion.astype(bool),
                }
            )
        )

    return pd.concat(frames, ignore_index=True).sort_values(["sku", "date"]).reset_index(drop=True)


def _promotion_blocks(rng: np.random.Generator, length: int) -> np.ndarray:
    """Mark roughly one week in six as promotional, in contiguous blocks."""
    flags = np.zeros(length, dtype=float)
    cursor = int(rng.integers(7, 28))
    while cursor < length:
        duration = int(rng.integers(3, 9))
        flags[cursor : cursor + duration] = 1.0
        cursor += duration + int(rng.integers(21, 56))
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (default: 7)")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    frame = generate(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, date_format="%Y-%m-%d")

    print(f"Wrote {len(frame):,} rows to {args.output}")
    print(f"  date range : {frame['date'].min():%Y-%m-%d} to {frame['date'].max():%Y-%m-%d}")
    print(f"  skus       : {', '.join(sorted(frame['sku'].unique()))}")
    print(f"  total units: {frame['units'].sum():,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
