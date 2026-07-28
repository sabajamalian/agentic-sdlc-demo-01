#!/usr/bin/env python3
"""Turn an approved Feature Proposals issue into Copilot-assigned issues.

This is the human-in-the-loop hop. It:

1. checks the comment is an ``/approve`` command
2. checks the commenter has write access
3. recovers the machine-readable payload from the proposals issue
4. creates one issue per approved proposal, assigned to the Copilot cloud agent
5. comments the results back and closes the proposals issue

Assignment uses ``copilot-swe-agent[bot]`` together with an ``agent_assignment``
body. That endpoint only accepts user-to-server tokens, so the workflow passes a
fine-grained PAT rather than the default GITHUB_TOKEN.

Usage:
    python scripts/approve_proposals.py --event "$GITHUB_EVENT_PATH"
    python scripts/approve_proposals.py --event event.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proposal_io import (
    ProposalError,
    parse_embedded_payload,
    parse_selection,
)

COPILOT_ASSIGNEE = "copilot-swe-agent[bot]"
PROPOSALS_LABEL = "feature-proposals"
WRITE_PERMISSIONS = {"admin", "write", "maintain"}

DEFAULT_LABELS = ["agent-generated"]


class ApprovalError(RuntimeError):
    """Raised when the approval cannot proceed."""


def gh_api(
    path: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    token: str | None = None,
) -> Any:
    """Call the GitHub REST API through the gh CLI."""
    command = ["gh", "api", "--method", method, path]
    if method != "GET":
        command += ["--input", "-"]

    environment = {**os.environ, "GH_TOKEN": token} if token else None

    completed = subprocess.run(
        command,
        input=json.dumps(body) if body is not None else None,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    if completed.returncode != 0:
        raise ApprovalError(
            f"gh api {method} {path} failed with exit {completed.returncode}:\n"
            f"{completed.stderr.strip()}"
        )

    if not completed.stdout.strip():
        return None
    return json.loads(completed.stdout)


def check_write_access(repo: str, login: str, token: str | None) -> str:
    """Raise unless ``login`` has write access to ``repo``."""
    try:
        response = gh_api(f"/repos/{repo}/collaborators/{login}/permission", token=token)
    except ApprovalError as error:
        raise ApprovalError(
            f"Could not verify permissions for @{login}. Treating this as a denial.\n{error}"
        ) from error

    permission = response.get("permission", "none")
    if permission not in WRITE_PERMISSIONS:
        raise ApprovalError(
            f"@{login} has '{permission}' access to {repo}. Approval requires write access."
        )
    return permission


def build_issue_body(
    proposal: dict[str, Any],
    payload: dict[str, Any],
    proposals_issue: int,
    number: int,
) -> str:
    """Render the issue body handed to the coding agent."""
    lines = [
        "## Problem",
        "",
        proposal["problem"],
        "",
        "## Acceptance criteria",
        "",
    ]
    lines += [f"- [ ] {criterion}" for criterion in proposal["acceptance_criteria"]]
    lines += ["- [ ] `make check` passes", ""]

    files = proposal.get("suggested_files") or []
    if files:
        lines += ["## Likely files", ""]
        lines += [f"- `{path}`" for path in files]
        lines.append("")

    if proposal.get("notes"):
        lines += ["## Notes and constraints", "", proposal["notes"], ""]

    if proposal.get("transcript_evidence"):
        lines += [
            "## Where this came from",
            "",
            f"> {proposal['transcript_evidence']}",
            "",
            f"Source: [`{payload['source_transcript']}`]"
            f"(../blob/HEAD/{payload['source_transcript']})",
            "",
        ]

    lines += [
        "## Working agreement",
        "",
        "Read `.github/copilot-instructions.md` before starting. In particular:",
        "",
        "- No random splits and no look-ahead. Lag anything derived from the target.",
        "- Add models through `src/forecasting/registry.py`, never by special-casing "
        "a script or notebook.",
        "- Do not loosen `eval/thresholds.yml` to make a build pass.",
        "- Strip notebook outputs before committing.",
        "",
        f"Approved as proposal {number} on #{proposals_issue}.",
    ]

    return "\n".join(lines)


def create_agent_issue(
    repo: str,
    title: str,
    body: str,
    labels: list[str],
    base_branch: str,
    token: str | None,
    assign_copilot: bool = True,
) -> dict[str, Any]:
    """Create an issue and hand it to the Copilot cloud agent."""
    payload: dict[str, Any] = {"title": title, "body": body, "labels": labels}

    if assign_copilot:
        payload["assignees"] = [COPILOT_ASSIGNEE]
        payload["agent_assignment"] = {
            "target_repo": repo,
            "base_branch": base_branch,
            "custom_instructions": "",
            "custom_agent": "",
            "model": "",
        }

    return gh_api(f"/repos/{repo}/issues", method="POST", body=payload, token=token)


def run(
    event: dict[str, Any],
    repo: str,
    base_branch: str,
    token: str | None,
    dry_run: bool = False,
    skip_permission_check: bool = False,
    assign_copilot: bool = True,
) -> dict[str, Any]:
    """Do the approval. Returns a summary dict."""
    issue = event.get("issue") or {}
    comment = event.get("comment") or {}

    labels = {label["name"] for label in issue.get("labels", [])}
    if PROPOSALS_LABEL not in labels:
        raise ApprovalError(
            f"Issue #{issue.get('number')} is not labelled '{PROPOSALS_LABEL}'. Ignoring."
        )

    commenter = (comment.get("user") or {}).get("login", "")
    body = comment.get("body", "")

    payload = parse_embedded_payload(issue.get("body") or "")
    proposals = payload["proposals"]
    if not proposals:
        raise ApprovalError("This proposals issue contains no proposals.")

    selection = parse_selection(body, len(proposals))

    if not skip_permission_check:
        permission = check_write_access(repo, commenter, token)
        print(f"@{commenter} has '{permission}' access, approval allowed")

    created: list[dict[str, Any]] = []
    for number in selection:
        proposal = proposals[number - 1]
        issue_labels = sorted(set(proposal.get("labels") or []) | set(DEFAULT_LABELS))
        issue_body = build_issue_body(proposal, payload, issue.get("number", 0), number)

        if dry_run:
            print(f"\n{'=' * 70}\n[{number}] {proposal['title']}")
            print(f"labels: {issue_labels}")
            print(f"assign: {COPILOT_ASSIGNEE if assign_copilot else '(none)'}")
            print(f"{'-' * 70}\n{issue_body}")
            created.append({"number": None, "title": proposal["title"], "html_url": "(dry run)"})
            continue

        result = create_agent_issue(
            repo=repo,
            title=proposal["title"],
            body=issue_body,
            labels=issue_labels,
            base_branch=base_branch,
            token=token,
            assign_copilot=assign_copilot,
        )
        print(f"created #{result['number']}: {result['html_url']}")
        created.append(result)

    return {
        "selected": selection,
        "total": len(proposals),
        "commenter": commenter,
        "created": created,
    }


def render_result_comment(summary: dict[str, Any], all_approved: bool) -> str:
    lines = [
        f"Approved by @{summary['commenter']}: "
        f"{len(summary['selected'])} of {summary['total']} proposals.",
        "",
    ]
    for number, issue in zip(summary["selected"], summary["created"], strict=True):
        reference = f"#{issue['number']}" if issue.get("number") else issue["html_url"]
        lines.append(f"- Proposal {number} -> {reference} {issue.get('title', '')}".rstrip())

    lines += [
        "",
        "Each issue is assigned to the Copilot cloud agent, which will open a draft "
        "pull request. Mark a pull request ready for review once you are happy with "
        "it; auto-merge takes over from there.",
    ]

    if not all_approved:
        remaining = summary["total"] - len(summary["selected"])
        lines += [
            "",
            f"{remaining} proposal(s) were not approved. Comment `/approve <numbers>` "
            "again to add them, or close this issue to drop them.",
        ]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True, help="Path to the event payload JSON")
    parser.add_argument("--repo", default=None, help="owner/repo (defaults to GITHUB_REPOSITORY)")
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--token", default=None, help="Token for gh (defaults to the environment)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print issues instead of creating them"
    )
    parser.add_argument(
        "--skip-permission-check",
        action="store_true",
        help="Skip the write-access check. Only for local testing.",
    )
    parser.add_argument(
        "--no-assign-copilot",
        action="store_true",
        help="Create the issues without handing them to the coding agent",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Append all_approved / created_count to this file (use $GITHUB_OUTPUT in CI)",
    )
    args = parser.parse_args()

    repo = args.repo or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("::error::--repo or GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2

    event = json.loads(args.event.read_text(encoding="utf-8"))

    try:
        summary = run(
            event=event,
            repo=repo,
            base_branch=args.base_branch,
            token=args.token,
            dry_run=args.dry_run,
            skip_permission_check=args.skip_permission_check,
            assign_copilot=not args.no_assign_copilot,
        )
    except (ApprovalError, ProposalError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    all_approved = len(summary["selected"]) == summary["total"]
    comment = render_result_comment(summary, all_approved)

    if args.summary_out:
        args.summary_out.write_text(comment + "\n", encoding="utf-8")

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"all_approved={'true' if all_approved else 'false'}\n")
            handle.write(f"created_count={len(summary['created'])}\n")

    print()
    print(comment)
    print()
    print(f"::notice::Created {len(summary['created'])} issue(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
