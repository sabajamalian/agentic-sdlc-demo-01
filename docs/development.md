# Development guide

## Prerequisites

- Python 3.11 or newer
- GNU Make
- Git

GitHub CLI is only needed to configure or operate the hosted demo. Local
forecasting development does not require network access after dependencies are
installed.

## Install the project

From the repository root:

```bash
make install
```

This creates `.venv`, upgrades `pip`, and installs the package in editable mode
with the `dev` optional dependencies. Make targets use executables from
`.venv/bin`, so activating the environment is optional.

To run Python commands directly, either activate the environment or use its
interpreter:

```bash
source .venv/bin/activate
# or
.venv/bin/python -c "import forecasting"
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/forecasting/` | Data loading, features, model implementations, registry, metrics, and backtesting |
| `scripts/` | Dataset generation, backtest and gate entry points, and agent-pipeline utilities |
| `tests/` | Unit and integration tests for forecasting and agent workflows |
| `notebooks/` | Executable EDA and model-comparison notebooks |
| `data/raw/sales.csv` | Deterministic generated dataset committed for offline use |
| `eval/thresholds.yml` | Accuracy and bias contract enforced by CI |
| `reports/` | Generated metrics, summaries, and plots; outputs are not source files |
| `.github/workflows/` | Transcript processing, approvals, reviews, CI, merging, and post-merge evaluation |
| `docs/` | Architecture, operation, forecasting, and contributor documentation |

## Common commands

```bash
make help              # list supported targets
make lint              # ruff checks and formatting verification
make test              # pytest suite
make notebooks         # execute notebooks from top to bottom
make backtest          # write reports/metrics.json
make gate              # run the backtest and enforce eval/thresholds.yml
make format            # apply ruff fixes and formatting
make strip             # remove notebook outputs
make clean             # remove caches and generated reports
make check             # run everything CI requires
```

Run `make check` before opening a pull request. It runs linting, verifies that
notebook outputs are absent, runs the test suite and notebooks, and evaluates
the forecasting gate.

## Development workflow

1. Read the relevant implementation and tests before editing.
2. Make the smallest complete change.
3. Add or update a behavioral test. A bug fix needs a regression test that fails
   without the fix.
4. Run the narrowest relevant test while iterating.
5. Run `make format`, then `make strip` if a notebook was touched.
6. Run `make check`.
7. Describe the behavior change, verification, and anything deliberately omitted
   in the pull request.

### Focused test runs

Pytest can target a file, class, or individual test through the virtual
environment:

```bash
.venv/bin/pytest tests/test_features.py
.venv/bin/pytest tests/test_models.py::TestSeasonalNaive
```

Do not remove or weaken unrelated tests to make a change pass.

## Time-series review checklist

Correctness depends on what information was available at forecast time:

- Use `train_test_split` or `rolling_origin_splits`; never randomly split a time
  series.
- Lag every feature derived from `units` by at least one observation.
- Only use an exogenous regressor when its future value is genuinely known.
  `on_promotion` and `price` meet that condition in this project.
- Build lag and rolling features on one series at a time. Call `select_sku` or
  `aggregate_total` before a feature builder.
- Add a leakage test for every new feature. Perturb a future target and verify
  that earlier feature values do not change.
- Do not raise evaluation thresholds merely to clear the gate.

See the [forecasting guide](forecasting.md) for the underlying contracts.

## Adding a model

A forecaster belongs in `src/forecasting/models/` and must:

1. subclass `Forecaster`;
2. provide a unique `name`;
3. implement `fit(history) -> self` and `predict(horizon) -> numpy.ndarray`;
4. be exported from `forecasting.models`;
5. be registered in `forecasting.registry`;
6. have contract and behavior tests.

Fast models belong in the `CHEAP_MODELS` parametrization in
`tests/test_models.py`. Slower models need focused tests marked
`@pytest.mark.slow`. Scripts and notebooks resolve models through the registry;
do not add model-specific branches to those callers.

Heavy dependencies belong in an optional dependency group. Modules for optional
models must remain importable when that extra is absent.

## Data and notebook rules

`data/raw/sales.csv` is produced by `scripts/generate_data.py` with seed 7.
Regenerate it with `make data`; do not edit the CSV manually. Changing the
generator or seed changes evaluation metrics and must be intentional.

Notebooks must execute without network access or manual steps and should finish
in about a minute. Never commit cell outputs:

```bash
make strip
make verify-notebooks
```

## Generated files

`make backtest` and `make gate` write files under `reports/`. Treat these as
local or CI artifacts unless a task explicitly requires them. Use `make clean`
to remove generated metrics, plots, and Python caches.

## Pull request expectations

Use the repository pull request template. State:

- what behavior or documentation changed;
- why the change is needed;
- which checks ran;
- before/after metrics when accuracy is affected;
- whether thresholds or dependencies changed;
- what was deliberately left out.

The template's time-series boxes may be marked not applicable for
documentation-only work, but the verification section should still report what
was checked.
