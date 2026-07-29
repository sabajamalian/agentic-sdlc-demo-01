# AGENTS.md

Conventions for any coding agent working in this repository.

The full instructions live in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md). Read that
file. The short version:

1. **Verify with `make check`** before opening a pull request. It runs lint, the
   notebook-output check, the test suite, notebook execution, and the model eval
   gate.
2. **No random splits, no look-ahead.** Use `rolling_origin_splits`. Lag anything
   derived from the target.
3. **Add models through `src/forecasting/registry.py`.** Never special-case a
   model in a script or a notebook.
4. **Do not loosen `eval/thresholds.yml`** to make a build pass.
5. **Never commit notebook outputs.** Run `make strip`.
6. **Do not hand-edit `data/raw/sales.csv`.** It is generated and verified.
7. **New runtime dependencies go in an optional extra**, not the default list.

This repository is a demo of an end-to-end agentic software development
lifecycle. Issues here are generated from meeting transcripts and assigned
automatically, and pull requests are reviewed by both a human and a review agent
before merge. Write pull request descriptions accordingly: state what changed,
what you verified, and what you deliberately left out.
