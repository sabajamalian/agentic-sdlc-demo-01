"""Tests for the agentic pipeline scripts.

These cover the parts of the workflow chain that can be verified without
GitHub: the JSON contract, the issue rendering, the ``/approve`` grammar, and
the permission guard. The GitHub calls themselves are stubbed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import approve_proposals
import proposal_io
import proposals_to_issue
import pytest
from agent_fixtures import issue_comment_event, sample_payload
from approve_proposals import (
    ApprovalError,
    build_issue_body,
    check_write_access,
    create_agent_issue,
    render_result_comment,
    run,
)
from proposal_io import (
    ProposalError,
    embed_payload,
    extract_json,
    load_payload,
    parse_embedded_payload,
    parse_selection,
    validate_payload,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "agent-prompts" / "proposals.schema.json"
)


@pytest.fixture
def payload() -> dict:
    return sample_payload()


@pytest.fixture
def issue_body(payload: dict) -> str:
    return proposals_to_issue.render_body(payload, run_url="https://example.invalid/run/1")


# --------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------


class TestExtractJson:
    def test_parses_bare_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_parses_fenced_json(self):
        text = 'Here you go:\n\n```json\n{"a": 1}\n```\n\nHope that helps.'
        assert extract_json(text) == {"a": 1}

    def test_parses_unlabelled_fence(self):
        assert extract_json('```\n{"a": [1, 2]}\n```') == {"a": [1, 2]}

    def test_falls_back_to_outermost_braces(self):
        assert extract_json('I made this: {"a": {"b": 2}} <- done') == {"a": {"b": 2}}

    def test_rejects_empty_output(self):
        with pytest.raises(ProposalError, match="empty"):
            extract_json("   \n  ")

    def test_rejects_prose_with_no_json(self):
        with pytest.raises(ProposalError, match="Could not find a JSON object"):
            extract_json("I was unable to complete this task.")

    def test_rejects_top_level_array(self):
        with pytest.raises(ProposalError, match="top level"):
            extract_json("[1, 2, 3]")

    def test_truncates_long_output_in_the_error(self):
        with pytest.raises(ProposalError, match=r"\.\.\."):
            extract_json("no json here " * 100)


class TestLoadPayload:
    def test_reads_a_file(self, tmp_path: Path, payload: dict):
        path = tmp_path / "proposals.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_payload(path)["meeting_date"] == "2026-07-15"

    def test_missing_file_names_the_path(self, tmp_path: Path):
        with pytest.raises(ProposalError, match="does not exist"):
            load_payload(tmp_path / "nope.json")


# --------------------------------------------------------------------------
# schema validation
# --------------------------------------------------------------------------


class TestValidatePayload:
    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists()
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_accepts_the_sample_payload(self, payload: dict):
        assert validate_payload(payload, SCHEMA_PATH) is payload

    def test_fills_optional_defaults(self):
        minimal = {
            "source_transcript": "docs/transcripts/x.md",
            "proposals": [
                {
                    "title": "Do the thing",
                    "problem": "The thing is not done, and it needs doing soon.",
                    "acceptance_criteria": ["it is done", "there is a test"],
                    "size": "small",
                }
            ],
        }
        result = validate_payload(minimal, SCHEMA_PATH)
        assert result["excluded"] == []
        assert result["proposals"][0]["suggested_files"] == []
        assert result["proposals"][0]["labels"] == []
        assert result["proposals"][0]["notes"] == ""

    def test_accepts_zero_proposals(self):
        payload = {"source_transcript": "docs/transcripts/x.md", "proposals": []}
        assert validate_payload(payload, SCHEMA_PATH)["proposals"] == []

    @pytest.mark.parametrize(
        ("mutation", "expected"),
        [
            (lambda p: p.pop("source_transcript"), "source_transcript"),
            (lambda p: p.pop("proposals"), "proposals"),
            (lambda p: p["proposals"][0].pop("title"), "title"),
            (lambda p: p["proposals"][0].pop("acceptance_criteria"), "acceptance_criteria"),
            (lambda p: p["proposals"][0].update(size="enormous"), "size"),
            (lambda p: p["proposals"][0].update(acceptance_criteria=["only one"]), "criteria"),
        ],
    )
    def test_rejects_malformed_payloads(self, payload: dict, mutation, expected: str):
        mutation(payload)
        with pytest.raises(ProposalError, match="schema validation"):
            validate_payload(payload, SCHEMA_PATH)

    def test_error_message_points_at_the_field(self, payload: dict):
        payload["proposals"][0]["size"] = "enormous"
        with pytest.raises(ProposalError) as info:
            validate_payload(payload, SCHEMA_PATH)
        assert "proposals/0/size" in str(info.value)


# --------------------------------------------------------------------------
# embed / recover round trip
# --------------------------------------------------------------------------


class TestEmbeddedPayload:
    def test_round_trips(self, payload: dict):
        assert parse_embedded_payload(embed_payload(payload)) == payload

    def test_round_trips_from_a_full_issue_body(self, payload: dict, issue_body: str):
        assert parse_embedded_payload(issue_body) == payload

    def test_survives_surrounding_prose(self, payload: dict):
        body = f"Some text\n\n{embed_payload(payload)}\n\nMore text"
        assert parse_embedded_payload(body) == payload

    def test_survives_an_earlier_json_fence(self, payload: dict):
        body = '```json\n{"decoy": true}\n```\n\n' + embed_payload(payload)
        assert parse_embedded_payload(body)["source_transcript"] == payload["source_transcript"]

    def test_preserves_non_ascii(self):
        payload = {"proposals": [], "note": "café, naïve, 日本語"}
        assert parse_embedded_payload(embed_payload(payload))["note"] == "café, naïve, 日本語"

    def test_missing_block_explains_why(self):
        with pytest.raises(ProposalError, match="no machine-readable proposal block"):
            parse_embedded_payload("Just a regular issue someone opened by hand.")

    def test_block_without_a_fence_is_rejected(self):
        body = f"{proposal_io.BEGIN_MARKER}\nno fence here\n{proposal_io.END_MARKER}"
        with pytest.raises(ProposalError, match="no JSON code fence"):
            parse_embedded_payload(body)

    def test_corrupted_json_is_rejected(self):
        body = f"{proposal_io.BEGIN_MARKER}\n```json\n{{not valid}}\n```\n{proposal_io.END_MARKER}"
        with pytest.raises(ProposalError, match="invalid"):
            parse_embedded_payload(body)

    def test_json_without_proposals_is_rejected(self):
        body = f'{proposal_io.BEGIN_MARKER}\n```json\n{{"a": 1}}\n```\n{proposal_io.END_MARKER}'
        with pytest.raises(ProposalError, match="missing a 'proposals' array"):
            parse_embedded_payload(body)


# --------------------------------------------------------------------------
# /approve grammar
# --------------------------------------------------------------------------


class TestParseSelection:
    @pytest.mark.parametrize(
        ("comment", "expected"),
        [
            ("/approve", [1, 2, 3]),
            ("/approve all", [1, 2, 3]),
            ("/approve ALL", [1, 2, 3]),
            ("/approve 2", [2]),
            ("/approve 1,3", [1, 3]),
            ("/approve 1, 3", [1, 3]),
            ("/approve 3 1", [1, 3]),
            ("/approve 1-3", [1, 2, 3]),
            ("/approve 2-3", [2, 3]),
            ("/approve 1,2-3", [1, 2, 3]),
            ("/approve 3,1,3", [1, 3]),
            ("/approve 2-2", [2]),
            ("  /approve 2  ", [2]),
            ("/approve 2.", [2]),
            ("/approve 1,2\n\nLooks good to me.", [1, 2]),
        ],
    )
    def test_accepted_forms(self, comment: str, expected: list[int]):
        assert parse_selection(comment, total=3) == expected

    @pytest.mark.parametrize(
        ("comment", "match"),
        [
            ("/approve 0", "out of range"),
            ("/approve 4", "out of range"),
            ("/approve 1-9", "out of range"),
            ("/approve 3-1", "start is greater"),
            ("/approve two", "Could not read"),
            ("/approve #2", "Could not read"),
            ("looks good", "Not an approval command"),
            ("", "Not an approval command"),
        ],
    )
    def test_rejected_forms(self, comment: str, match: str):
        with pytest.raises(ProposalError, match=match):
            parse_selection(comment, total=3)

    def test_only_the_first_line_is_a_command(self):
        assert parse_selection("/approve 1\n/approve 2", total=3) == [1]


# --------------------------------------------------------------------------
# issue rendering
# --------------------------------------------------------------------------


class TestRenderProposalsIssue:
    def test_title_counts_proposals(self, payload: dict):
        assert proposals_to_issue.render_title(payload) == (
            "Feature proposals: Forecasting working session (3 proposals)"
        )

    def test_title_is_singular_for_one(self, payload: dict):
        payload["proposals"] = payload["proposals"][:1]
        assert proposals_to_issue.render_title(payload).endswith("(1 proposal)")

    def test_title_falls_back_to_the_transcript_name(self, payload: dict):
        payload.pop("meeting_title")
        assert "2026-07-15-forecasting-planning" in proposals_to_issue.render_title(payload)

    def test_body_lists_every_proposal(self, payload: dict, issue_body: str):
        for index, proposal in enumerate(payload["proposals"], start=1):
            assert f"### {index}. {proposal['title']}" in issue_body

    def test_body_documents_the_approval_commands(self, issue_body: str):
        assert "`/approve`" in issue_body
        assert "`/approve 1,3`" in issue_body
        assert "`/approve 2-4`" in issue_body

    def test_body_includes_excluded_topics(self, issue_body: str):
        assert "Discussed but not proposed" in issue_body
        assert "Weather features" in issue_body

    def test_body_escapes_pipes_in_the_excluded_table(self, payload: dict):
        payload["excluded"] = [{"topic": "a|b", "reason": "c|d"}]
        body = proposals_to_issue.render_body(payload)
        assert "a\\|b" in body

    def test_body_links_the_workflow_run(self, issue_body: str):
        assert "https://example.invalid/run/1" in issue_body

    def test_empty_proposals_still_render(self, payload: dict):
        payload["proposals"] = []
        body = proposals_to_issue.render_body(payload)
        assert "found no actionable work" in body
        assert parse_embedded_payload(body)["proposals"] == []

    def test_cli_writes_body_and_title(self, tmp_path: Path, payload: dict, monkeypatch):
        proposals = tmp_path / "proposals.json"
        proposals.write_text(json.dumps(payload), encoding="utf-8")
        body_out = tmp_path / "body.md"
        title_out = tmp_path / "title.txt"

        monkeypatch.setattr(
            "sys.argv",
            [
                "proposals_to_issue.py",
                "--proposals",
                str(proposals),
                "--body-out",
                str(body_out),
                "--title-out",
                str(title_out),
            ],
        )
        assert proposals_to_issue.main() == 0
        assert "Feature proposals" in title_out.read_text(encoding="utf-8")
        assert parse_embedded_payload(body_out.read_text(encoding="utf-8")) == payload

    def test_cli_fails_cleanly_on_bad_agent_output(self, tmp_path: Path, monkeypatch, capsys):
        proposals = tmp_path / "proposals.json"
        proposals.write_text("I could not do it, sorry.", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "proposals_to_issue.py",
                "--proposals",
                str(proposals),
                "--body-out",
                str(tmp_path / "body.md"),
                "--title-out",
                str(tmp_path / "title.txt"),
            ],
        )
        assert proposals_to_issue.main() == 1
        assert "::error::" in capsys.readouterr().err

    def test_cli_can_override_the_transcript_path(self, tmp_path: Path, payload: dict, monkeypatch):
        proposals = tmp_path / "proposals.json"
        payload["source_transcript"] = "wrong/path.md"
        proposals.write_text(json.dumps(payload), encoding="utf-8")
        body_out = tmp_path / "body.md"

        monkeypatch.setattr(
            "sys.argv",
            [
                "proposals_to_issue.py",
                "--proposals",
                str(proposals),
                "--body-out",
                str(body_out),
                "--title-out",
                str(tmp_path / "title.txt"),
                "--source-transcript",
                "docs/transcripts/right.md",
            ],
        )
        assert proposals_to_issue.main() == 0
        recovered = parse_embedded_payload(body_out.read_text(encoding="utf-8"))
        assert recovered["source_transcript"] == "docs/transcripts/right.md"


# --------------------------------------------------------------------------
# agent issue body
# --------------------------------------------------------------------------


class TestBuildIssueBody:
    def test_contains_the_acceptance_criteria_as_checkboxes(self, payload: dict):
        proposal = payload["proposals"][0]
        body = build_issue_body(proposal, payload, proposals_issue=42, number=1)
        for criterion in proposal["acceptance_criteria"]:
            assert f"- [ ] {criterion}" in body

    def test_always_requires_make_check(self, payload: dict):
        body = build_issue_body(payload["proposals"][0], payload, 42, 1)
        assert "- [ ] `make check` passes" in body

    def test_references_the_proposals_issue(self, payload: dict):
        body = build_issue_body(payload["proposals"][0], payload, 42, 1)
        assert "proposal 1 on #42" in body

    def test_repeats_the_guardrails(self, payload: dict):
        body = build_issue_body(payload["proposals"][0], payload, 42, 1)
        assert "copilot-instructions.md" in body
        assert "look-ahead" in body
        assert "eval/thresholds.yml" in body

    def test_omits_optional_sections_when_absent(self, payload: dict):
        proposal = payload["proposals"][0]
        proposal["suggested_files"] = []
        proposal["notes"] = ""
        proposal["transcript_evidence"] = ""
        body = build_issue_body(proposal, payload, 42, 1)
        assert "## Likely files" not in body
        assert "## Notes and constraints" not in body
        assert "## Where this came from" not in body


# --------------------------------------------------------------------------
# approval flow, with GitHub stubbed out
# --------------------------------------------------------------------------


class FakeGitHub:
    """Records gh_api calls and returns plausible responses."""

    def __init__(self, permission: str = "write"):
        self.permission = permission
        self.calls: list[tuple[str, str, dict | None]] = []
        self._next_issue = 100

    def __call__(self, path, method="GET", body=None, token=None):
        self.calls.append((method, path, body))
        if "/permission" in path:
            if self.permission == "denied":
                raise ApprovalError("HTTP 404: Not Found")
            return {"permission": self.permission}
        if path.endswith("/issues") and method == "POST":
            self._next_issue += 1
            return {
                "number": self._next_issue,
                "title": body["title"],
                "html_url": f"https://github.com/o/r/issues/{self._next_issue}",
            }
        raise AssertionError(f"unexpected call: {method} {path}")

    @property
    def created_issues(self) -> list[dict]:
        return [body for method, path, body in self.calls if method == "POST" and body]


@pytest.fixture
def fake_github(monkeypatch) -> FakeGitHub:
    fake = FakeGitHub()
    monkeypatch.setattr(approve_proposals, "gh_api", fake)
    return fake


class TestRun:
    def test_creates_one_issue_per_proposal(self, issue_body: str, fake_github: FakeGitHub):
        event = issue_comment_event("/approve", issue_body)
        summary = run(event, repo="o/r", base_branch="main", token="t")

        assert summary["selected"] == [1, 2, 3]
        assert summary["total"] == 3
        assert len(fake_github.created_issues) == 3

    def test_assigns_the_coding_agent(self, issue_body: str, fake_github: FakeGitHub):
        run(
            issue_comment_event("/approve 1", issue_body), repo="o/r", base_branch="main", token="t"
        )

        created = fake_github.created_issues[0]
        assert created["assignees"] == ["copilot-swe-agent[bot]"]
        assert created["agent_assignment"]["target_repo"] == "o/r"
        assert created["agent_assignment"]["base_branch"] == "main"

    def test_honours_a_non_default_base_branch(self, issue_body: str, fake_github: FakeGitHub):
        run(issue_comment_event("/approve 1", issue_body), repo="o/r", base_branch="dev", token="t")
        assert fake_github.created_issues[0]["agent_assignment"]["base_branch"] == "dev"

    def test_can_skip_the_agent_assignment(self, issue_body: str, fake_github: FakeGitHub):
        run(
            issue_comment_event("/approve 1", issue_body),
            repo="o/r",
            base_branch="main",
            token="t",
            assign_copilot=False,
        )
        created = fake_github.created_issues[0]
        assert "assignees" not in created
        assert "agent_assignment" not in created

    def test_always_applies_the_agent_generated_label(
        self, issue_body: str, fake_github: FakeGitHub
    ):
        run(
            issue_comment_event("/approve 1", issue_body), repo="o/r", base_branch="main", token="t"
        )
        assert "agent-generated" in fake_github.created_issues[0]["labels"]

    def test_keeps_the_proposal_labels(self, issue_body: str, fake_github: FakeGitHub):
        run(
            issue_comment_event("/approve 1", issue_body), repo="o/r", base_branch="main", token="t"
        )
        assert "forecasting" in fake_github.created_issues[0]["labels"]

    def test_partial_selection_creates_a_subset(self, issue_body: str, fake_github: FakeGitHub):
        summary = run(
            issue_comment_event("/approve 1,3", issue_body),
            repo="o/r",
            base_branch="main",
            token="t",
        )
        assert summary["selected"] == [1, 3]
        assert len(fake_github.created_issues) == 2

    def test_checks_permissions_first(self, issue_body: str, fake_github: FakeGitHub):
        run(issue_comment_event("/approve", issue_body), repo="o/r", base_branch="main", token="t")
        assert "/permission" in fake_github.calls[0][1]

    def test_rejects_a_commenter_without_write_access(self, issue_body: str, monkeypatch):
        monkeypatch.setattr(approve_proposals, "gh_api", FakeGitHub(permission="read"))
        with pytest.raises(ApprovalError, match="requires write access"):
            run(
                issue_comment_event("/approve", issue_body),
                repo="o/r",
                base_branch="main",
                token="t",
            )

    def test_treats_an_unverifiable_permission_as_denial(self, issue_body: str, monkeypatch):
        monkeypatch.setattr(approve_proposals, "gh_api", FakeGitHub(permission="denied"))
        with pytest.raises(ApprovalError, match="Treating this as a denial"):
            run(
                issue_comment_event("/approve", issue_body),
                repo="o/r",
                base_branch="main",
                token="t",
            )

    def test_creates_nothing_when_permission_is_denied(self, issue_body: str, monkeypatch):
        fake = FakeGitHub(permission="read")
        monkeypatch.setattr(approve_proposals, "gh_api", fake)
        with pytest.raises(ApprovalError):
            run(
                issue_comment_event("/approve", issue_body),
                repo="o/r",
                base_branch="main",
                token="t",
            )
        assert fake.created_issues == []

    def test_ignores_issues_without_the_label(self, issue_body: str, fake_github: FakeGitHub):
        event = issue_comment_event("/approve", issue_body, labels=["bug"])
        with pytest.raises(ApprovalError, match="not labelled"):
            run(event, repo="o/r", base_branch="main", token="t")

    def test_rejects_an_issue_with_no_embedded_payload(self, fake_github: FakeGitHub):
        event = issue_comment_event("/approve", "someone opened this by hand")
        with pytest.raises(ProposalError, match="no machine-readable"):
            run(event, repo="o/r", base_branch="main", token="t")

    def test_dry_run_creates_nothing(self, issue_body: str, fake_github: FakeGitHub, capsys):
        summary = run(
            issue_comment_event("/approve", issue_body),
            repo="o/r",
            base_branch="main",
            token="t",
            dry_run=True,
            skip_permission_check=True,
        )
        assert fake_github.calls == []
        assert len(summary["created"]) == 3
        assert "Add a Prophet forecaster" in capsys.readouterr().out


class TestResultComment:
    def test_lists_every_created_issue(self, issue_body: str, fake_github: FakeGitHub):
        summary = run(
            issue_comment_event("/approve", issue_body), repo="o/r", base_branch="main", token="t"
        )
        comment = render_result_comment(summary, all_approved=True)
        assert "Proposal 1 -> #101" in comment
        assert "3 of 3 proposals" in comment
        assert "not approved" not in comment

    def test_mentions_the_remainder_on_a_partial_approval(
        self, issue_body: str, fake_github: FakeGitHub
    ):
        summary = run(
            issue_comment_event("/approve 2", issue_body),
            repo="o/r",
            base_branch="main",
            token="t",
        )
        comment = render_result_comment(summary, all_approved=False)
        assert "1 of 3 proposals" in comment
        assert "2 proposal(s) were not approved" in comment

    def test_credits_the_approver(self, issue_body: str, fake_github: FakeGitHub):
        summary = run(
            issue_comment_event("/approve", issue_body, login="dana"),
            repo="o/r",
            base_branch="main",
            token="t",
        )
        assert "@dana" in render_result_comment(summary, all_approved=True)


class TestApproveCli:
    def test_dry_run_end_to_end(self, tmp_path: Path, issue_body: str, monkeypatch):
        event_path = tmp_path / "event.json"
        event_path.write_text(
            json.dumps(issue_comment_event("/approve 1,3", issue_body)), encoding="utf-8"
        )
        summary_out = tmp_path / "summary.md"
        output = tmp_path / "gh_output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "approve_proposals.py",
                "--event",
                str(event_path),
                "--repo",
                "o/r",
                "--dry-run",
                "--skip-permission-check",
                "--summary-out",
                str(summary_out),
                "--github-output",
                str(output),
            ],
        )
        assert approve_proposals.main() == 0
        assert "2 of 3 proposals" in summary_out.read_text(encoding="utf-8")
        assert "all_approved=false" in output.read_text(encoding="utf-8")
        assert "created_count=2" in output.read_text(encoding="utf-8")

    def test_reports_all_approved(self, tmp_path: Path, issue_body: str, monkeypatch):
        event_path = tmp_path / "event.json"
        event_path.write_text(
            json.dumps(issue_comment_event("/approve", issue_body)), encoding="utf-8"
        )
        output = tmp_path / "gh_output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "approve_proposals.py",
                "--event",
                str(event_path),
                "--repo",
                "o/r",
                "--dry-run",
                "--skip-permission-check",
                "--github-output",
                str(output),
            ],
        )
        assert approve_proposals.main() == 0
        assert "all_approved=true" in output.read_text(encoding="utf-8")

    def test_failure_exits_non_zero(self, tmp_path: Path, issue_body: str, monkeypatch, capsys):
        event_path = tmp_path / "event.json"
        event_path.write_text(
            json.dumps(issue_comment_event("/approve 9", issue_body)), encoding="utf-8"
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "approve_proposals.py",
                "--event",
                str(event_path),
                "--repo",
                "o/r",
                "--dry-run",
                "--skip-permission-check",
            ],
        )
        assert approve_proposals.main() == 1
        assert "out of range" in capsys.readouterr().err

    def test_repo_is_required(self, tmp_path: Path, issue_body: str, monkeypatch, capsys):
        event_path = tmp_path / "event.json"
        event_path.write_text(
            json.dumps(issue_comment_event("/approve", issue_body)), encoding="utf-8"
        )
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.setattr(
            "sys.argv", ["approve_proposals.py", "--event", str(event_path), "--dry-run"]
        )
        assert approve_proposals.main() == 2
        assert "GITHUB_REPOSITORY" in capsys.readouterr().err


# --------------------------------------------------------------------------
# gh_api plumbing
# --------------------------------------------------------------------------


class TestGhApi:
    def test_raises_with_stderr_on_failure(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="HTTP 403: Forbidden")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ApprovalError, match="HTTP 403"):
            approve_proposals.gh_api("/repos/o/r/issues")

    def test_returns_none_on_empty_output(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert approve_proposals.gh_api("/repos/o/r/issues") is None

    def test_passes_the_token_through_the_environment(self, monkeypatch):
        seen = {}

        def fake_run(command, **kwargs):
            seen["env"] = kwargs.get("env")
            seen["command"] = command
            seen["input"] = kwargs.get("input")
            return subprocess.CompletedProcess(command, 0, stdout='{"ok": true}', stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        approve_proposals.gh_api("/x", method="POST", body={"a": 1}, token="secret")

        assert seen["env"]["GH_TOKEN"] == "secret"
        assert "--input" in seen["command"]
        assert json.loads(seen["input"]) == {"a": 1}

    def test_get_requests_send_no_body(self, monkeypatch):
        seen = {}

        def fake_run(command, **kwargs):
            seen["command"] = command
            return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        approve_proposals.gh_api("/x")
        assert "--input" not in seen["command"]


class TestCheckWriteAccess:
    @pytest.mark.parametrize("permission", ["admin", "write", "maintain"])
    def test_allows_write_roles(self, permission: str, monkeypatch):
        monkeypatch.setattr(approve_proposals, "gh_api", FakeGitHub(permission=permission))
        assert check_write_access("o/r", "someone", None) == permission

    @pytest.mark.parametrize("permission", ["read", "none", "triage"])
    def test_blocks_everything_else(self, permission: str, monkeypatch):
        monkeypatch.setattr(approve_proposals, "gh_api", FakeGitHub(permission=permission))
        with pytest.raises(ApprovalError, match="requires write access"):
            check_write_access("o/r", "someone", None)


class TestCreateAgentIssue:
    def test_posts_to_the_issues_endpoint(self, monkeypatch):
        fake = FakeGitHub()
        monkeypatch.setattr(approve_proposals, "gh_api", fake)
        create_agent_issue("o/r", "t", "b", ["x"], "main", None)
        method, path, _ = fake.calls[0]
        assert (method, path) == ("POST", "/repos/o/r/issues")
