# Documentation

This directory explains how to run, extend, and operate the agentic SDLC demo
and its demand-forecasting project.

## Start here

| Guide | Audience | Contents |
| --- | --- | --- |
| [Development guide](development.md) | Contributors | Local setup, repository layout, common commands, validation, and contribution workflow |
| [Forecasting guide](forecasting.md) | Data scientists and model developers | Data contracts, leakage-safe features, model interfaces, backtesting, metrics, and the evaluation gate |
| [Architecture](architecture.md) | Maintainers and reviewers | Agent workflow, human gates, state transfer, token design, and failure handling |
| [Demo runbook](runbook.md) | Demo operators | One-time GitHub setup, live-demo steps, troubleshooting, cost, and reset instructions |
| [Planning transcript](transcripts/2026-07-15-forecasting-planning.md) | Demo participants | Example input that starts the transcript-to-proposal workflow |

The root [README](../README.md) gives the shortest project overview and
quickstart. The guides here provide the details needed to make or operate
changes safely.

## Documentation map

The project has two connected systems:

1. **Demand forecasting.** Python code loads deterministic synthetic sales data,
   creates leakage-safe features, fits registered forecasters, and evaluates
   them with rolling-origin backtests.
2. **Agentic delivery.** GitHub Actions turns meeting transcripts into proposals,
   applies two human approval gates, dispatches coding agents, and runs
   automated review and evaluation.

Use the [forecasting guide](forecasting.md) when changing product behavior. Use
the [architecture](architecture.md) and [runbook](runbook.md) when changing or
operating the delivery pipeline. All contributors should follow the
[development guide](development.md).

## Keeping these docs current

Update the relevant guide whenever a change alters:

- setup commands, supported Python versions, or dependencies;
- a public data, feature, model, or evaluation contract;
- workflow triggers, permissions, secrets, labels, or human gates;
- generated artifacts or troubleshooting steps.

Documentation examples must use chronological time-series splits and must not
suggest weakening `eval/thresholds.yml` to pass CI.
