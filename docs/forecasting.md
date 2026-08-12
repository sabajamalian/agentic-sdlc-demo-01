# Forecasting guide

This guide describes the forecasting contracts that implementation changes
must preserve.

## Data flow

The default evaluation path is:

```text
data/raw/sales.csv
  -> load_sales
  -> aggregate_total or select_sku
  -> rolling_origin_splits
  -> registry model fitted independently at each origin
  -> pooled and per-fold metrics
  -> reports/metrics.json
  -> eval/thresholds.yml checks
```

The committed dataset is synthetic and deterministic so local runs, CI, and
agent-authored changes evaluate identical observations. See
[`data/README.md`](../data/README.md) for its schema and generating process.

## Data contracts

`load_sales` returns rows sorted by `sku` and `date`. The required columns are:

| Column | Meaning | Forecast-time availability |
| --- | --- | --- |
| `date` | Daily observation date | Known |
| `sku` | Series identifier | Known |
| `units` | Demand target | Historical values only |
| `price` | Planned selling price | Known in advance for this dataset |
| `on_promotion` | Planned promotion flag | Known in advance for this dataset |

Functions that shift, roll, split, or fit along time expect a single series.
Create one with:

- `select_sku(frame, sku)` for one product; or
- `aggregate_total(frame)` for total demand.

`validate_single_series` rejects multiple SKUs and duplicate dates. This guard
prevents a shift over interleaved products from silently using another SKU's
value as a lag.

## Chronological splitting

`train_test_split(frame, horizon)` reserves the final `horizon` rows for testing.

`rolling_origin_splits(frame, horizon, n_splits, step, min_train_size)` creates
expanding training windows in chronological order. Each test window begins
immediately after its training cutoff. By default, `step` equals `horizon`, so
test windows do not overlap.

For each rolling origin, `backtest` creates and fits a fresh registered model.
That prevents fitted state from one fold leaking into another.

Random train/test splits are invalid for this project because they expose later
observations during training.

## Feature engineering

`build_feature_frame` sorts a single series and applies the standard feature
pipeline:

1. calendar fields such as day of week and month;
2. Fourier terms anchored to a fixed epoch;
3. target lags;
4. target rolling means and standard deviations;
5. promotion features when `on_promotion` is present;
6. optional removal of warm-up rows containing missing values.

### Leakage rules

Features fall into two categories:

- **Known in advance:** calendar values, fixed-epoch Fourier terms, planned
  promotion state, planned price.
- **Derived from the target:** lags and rolling statistics based on `units`.

Target-derived features must be shifted by at least one row. Rolling windows
shift first and aggregate second, so a row never uses its own target. Promotion
features use the schedule and do not inspect `units`.

When adding a feature, tests should perturb a target near the end of the series
and assert that earlier feature rows remain unchanged. Date encodings should
also produce the same value for the same date regardless of the input slice.

`feature_columns` returns numeric and boolean model inputs while excluding
`date`, `sku`, and the target.

## Model contract and registry

Every model subclasses `forecasting.models.base.Forecaster`.

`fit(history)`:

- receives one chronologically sorted series;
- rejects empty, non-finite, or otherwise insufficient target history;
- reads only rows supplied in `history`;
- returns the fitted instance.

`predict(horizon)`:

- requires a fitted model;
- rejects non-positive horizons;
- returns exactly `horizon` finite point forecasts as a NumPy array.

The built-in registry names are:

| Name | Implementation | Behavior |
| --- | --- | --- |
| `seasonal_naive` | `SeasonalNaiveForecaster` | Repeats the last season; default baseline uses seven days |
| `mean` | `MeanForecaster` | Repeats the full-history or trailing-window mean |
| `sarimax` | `SarimaxForecaster` | Fits a compact weekly seasonal ARIMA model |

`get_model(name, **params)` returns a new instance. Callers use registry names
so evaluation can construct an independent model for every fold.

## Backtesting and metrics

`backtest` calculates metrics for each fold and again over all pooled
predictions. Optional prediction collection returns fold, date, actual, and
predicted values.

| Metric | Interpretation |
| --- | --- |
| MAE | Mean absolute error in target units |
| RMSE | Root mean squared error; weights larger misses more heavily |
| MAPE | Mean absolute percentage error; zero actuals are excluded |
| sMAPE | Symmetric percentage error, bounded from 0 to 2 |
| Bias | Mean signed error; positive values indicate over-forecasting |

`scripts/run_backtest.py` evaluates the default model set on `TOTAL` and includes
per-SKU metrics for every model. It writes configuration, dataset metadata,
aggregate results, folds, and per-SKU rows to `reports/metrics.json`.

Useful exploratory commands:

```bash
make backtest
.venv/bin/python scripts/run_backtest.py --horizon 7 --n-splits 8 --plot
.venv/bin/python scripts/run_backtest.py --series SKU-ALPHA
```

## Evaluation gate

`make gate` first regenerates `reports/metrics.json`, then
`scripts/check_eval_gate.py` compares the champion against
`eval/thresholds.yml`. The contract checks:

- maximum champion MAPE, MAE, and RMSE;
- maximum absolute bias;
- minimum number of rolling origins;
- minimum relative MAPE improvement over the seasonal naive baseline.

The checker also writes `reports/eval_summary.md`, which CI can post on a pull
request. A threshold change is a product decision, not a workaround for a
regression. Any justified threshold change must include before/after values and
an explanation in the pull request.

## Safe extension checklist

Before merging a forecasting change:

- the input is one series and chronologically ordered;
- no target-derived feature reads the present or future;
- future exogenous values are genuinely available at prediction time;
- a new model follows the base contract and is registered;
- backtest folds fit fresh model instances;
- behavior and leakage tests cover the change;
- `eval/thresholds.yml` was not loosened to hide a regression;
- `make check` passes.
