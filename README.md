# Agentic SDLC demo

A working demonstration of a software development lifecycle where agents do the
work and humans make two decisions.

```
meeting transcript
  -> agent parses it into feature proposals
  -> HUMAN: /approve on a proposals issue
  -> issues created, assigned to the Copilot cloud agent
  -> agent opens draft pull requests
  -> Copilot code review + a data science review agent
  -> CI: lint, tests, notebook execution, model eval gate
  -> HUMAN: mark the pull request ready for review
  -> squash auto-merge
  -> post-merge backtest on main
```

The product being built is a demand forecasting project: pandas, statsmodels
SARIMAX, a seasonal naive baseline, notebooks, and a MAPE gate in CI. Real data
science work, so CI has something meaningful to gate on and the reviews have
something real to catch.

See [`docs/architecture.md`](docs/architecture.md) for the pipeline diagram and
the design reasoning, and [`docs/runbook.md`](docs/runbook.md) for setup and the
live demo script.

## Quickstart

```bash
make install     # venv + dev dependencies
make check       # lint, 239 tests, notebook execution, eval gate
```

Individual targets:

```bash
make data        # regenerate the synthetic dataset
make test        # unit tests only
make notebooks   # execute every notebook, fail on error
make backtest    # rolling-origin backtest -> reports/metrics.json
make gate        # backtest, then check against eval/thresholds.yml
make format      # ruff fix + format
```

Python 3.11 or newer.

## Running the demo

Setup is six steps, three of them required. Full detail in
[`docs/runbook.md`](docs/runbook.md); the short version:

```bash
gh secret set AGENT_PAT      # fine-grained PAT, see the runbook for scopes
scripts/bootstrap_repo.sh    # create labels, check the setup
```

Then enable the Copilot cloud agent under **Settings → Copilot → Cloud agent**,
including *Allow GitHub Actions workflows to run without approval*.

Kick it off:

```bash
git add docs/transcripts/2026-07-15-forecasting-planning.md
git commit -m "Add forecasting planning transcript"
git push
```

A "Feature proposals" issue appears within a few minutes. Comment `/approve`, or
`/approve 1,3` to take a subset.

## What is in here

### The pipeline

| Workflow | Trigger | What it does |
|---|---|---|
| `01-transcript-to-proposals.yml` | push to `docs/transcripts/**.md` | Copilot CLI turns the transcript into schema-validated JSON; Python renders and opens the proposals issue |
| `02-approve-proposals.yml` | `/approve` comment | Verifies write access, creates one issue per approved proposal, assigns the coding agent |
| `03-pr-review-agents.yml` | pull request events, plus a sweeper | Requests Copilot code review and runs a data science reviewer |
| `04-auto-merge.yml` | ready for review, check suite | Enables squash auto-merge on eligible `copilot/*` pull requests |
| `ci.yml` | pull request, push to main | Lint, tests, notebook execution, eval gate with a metrics comment |
| `post-merge-eval.yml` | push to main, weekly | Full backtest, metrics and plots as artifacts |

### The forecasting project

```
src/forecasting/
  data.py        loading, SKU selection, rolling-origin splits
  features.py    calendar, lag and rolling features, all shifted
  models/        Forecaster protocol, seasonal naive, SARIMAX
  registry.py    name -> factory. Adding a model is one line
  evaluate.py    MAE / RMSE / MAPE / sMAPE / bias, rolling-origin backtest

scripts/
  generate_data.py     regenerate data/raw/sales.csv (seeded)
  run_backtest.py      -> reports/metrics.json
  check_eval_gate.py   compare against eval/thresholds.yml, exit 1 on regression
  proposals_to_issue.py, approve_proposals.py, proposal_io.py
  bootstrap_repo.sh

notebooks/       01-eda.ipynb, 02-baseline-model.ipynb (outputs stripped, CI enforces it)
tests/           239 tests covering the library, the scripts and the agent plumbing
```

Current numbers on the committed dataset: SARIMAX at **9.32% MAPE** against a
seasonal naive baseline of **10.32%**, over 5 rolling origins at a 14-day
horizon, aggregate series.

## Two design choices worth knowing about

**Agents write files, not API calls.** Every agent step writes to disk and
deterministic Python takes it from there. The transcript agent's output is
validated against
[`.github/agent-prompts/proposals.schema.json`](.github/agent-prompts/proposals.schema.json)
before anything is created, so a bad model response fails with a message naming
the field rather than opening a malformed issue. The workflow YAML stays thin and
the logic gets unit tests.

**The gaps are deliberate.** There is no Prophet model, nothing reads the
`on_promotion` column, and evaluation is aggregate-only. The sample transcript
asks for exactly those three things, so the coding agent builds functionality
that is genuinely missing.

## Requirements

- Python 3.11+
- A GitHub repository with Copilot coding agent access
- A fine-grained PAT stored as `AGENT_PAT` (scopes in the runbook)
- Node 22+ in the runner, for Copilot CLI

Each CLI invocation and coding agent session consumes premium requests. Costs are
broken down in the runbook.
