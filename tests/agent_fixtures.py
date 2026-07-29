"""A realistic agent-produced proposals payload, used across the agent tests."""

from __future__ import annotations

from typing import Any


def sample_payload() -> dict[str, Any]:
    """Return a fresh copy of a schema-valid proposals payload."""
    return {
        "source_transcript": "docs/transcripts/2026-07-15-forecasting-planning.md",
        "meeting_title": "Forecasting working session",
        "meeting_date": "2026-07-15",
        "proposals": [
            {
                "title": "Add per-SKU backtest reporting alongside the aggregate metrics",
                "problem": (
                    "The backtest only evaluates the TOTAL series, so per-SKU errors "
                    "cancel out and replenishment decisions are made on a number that "
                    "hides the variance."
                ),
                "acceptance_criteria": [
                    "run_backtest.py evaluates every SKU as well as the aggregate",
                    "reports/metrics.json contains a per_sku section",
                    "The report sorts SKUs worst MAPE first",
                ],
                "suggested_files": [
                    "scripts/run_backtest.py",
                    "src/forecasting/evaluate.py",
                ],
                "labels": ["forecasting", "evaluation"],
                "size": "medium",
                "notes": "Keep the aggregate metrics; this is additive.",
                "transcript_evidence": (
                    "The aggregate number looks fine and then I go to actually order "
                    "for a specific SKU and it's nowhere near."
                ),
            },
            {
                "title": "Add a Prophet forecaster behind the model registry",
                "problem": (
                    "SARIMAX under-forecasts the holiday ramp. Prophet handles "
                    "multiple seasonalities and changepoints and should be evaluated "
                    "as a candidate champion."
                ),
                "acceptance_criteria": [
                    "ProphetForecaster implements the Forecaster protocol",
                    "The model is registered in src/forecasting/registry.py",
                    "prophet stays an optional extra and is imported lazily",
                    "The pull request reports a backtest against SARIMAX and the baseline",
                ],
                "suggested_files": [
                    "src/forecasting/models/prophet_model.py",
                    "src/forecasting/registry.py",
                ],
                "labels": ["forecasting", "model"],
                "size": "large",
                "notes": "CI must still install and pass without the extra.",
                "transcript_evidence": "I don't want to add a dependency on vibes.",
            },
            {
                "title": "Add holiday and promotion calendar features",
                "problem": (
                    "The models ignore the on_promotion flag and have no holiday "
                    "calendar, so promotional weekends and public holidays are "
                    "forecast as ordinary days."
                ),
                "acceptance_criteria": [
                    "US federal holiday flags are available as features",
                    "Promotion features are derived from the existing on_promotion column",
                    "A look-ahead test covers every new feature",
                ],
                "suggested_files": ["src/forecasting/features.py"],
                "labels": ["forecasting", "features"],
                "size": "medium",
                "notes": "US holidays only. Anything derived from historical units "
                "must be fitted on the training window.",
                "transcript_evidence": "We have the on_promotion flag in the raw data. "
                "Nothing uses it.",
            },
        ],
        "excluded": [
            {
                "topic": "Weather features",
                "reason": "Needs a forecast weather feed; parked for this quarter.",
            },
            {
                "topic": "Retraining cadence",
                "reason": "Unresolved. Revisit after the per-SKU numbers land.",
            },
            {
                "topic": "Moving training off the shared runner",
                "reason": "Platform work, tracked in the platform sync.",
            },
        ],
    }


def issue_comment_event(
    comment_body: str,
    issue_body: str,
    *,
    issue_number: int = 42,
    login: str = "sabajamalian",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal ``issue_comment`` webhook payload."""
    if labels is None:
        labels = ["feature-proposals"]
    return {
        "action": "created",
        "issue": {
            "number": issue_number,
            "title": "Feature proposals: Forecasting working session (3 proposals)",
            "state": "open",
            "body": issue_body,
            "labels": [{"name": name} for name in labels],
        },
        "comment": {
            "id": 987654,
            "body": comment_body,
            "user": {"login": login},
        },
        "repository": {"full_name": "sabajamalian/agentic-sdlc-demo-01"},
    }
