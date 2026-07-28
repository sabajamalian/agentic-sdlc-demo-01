"""Shared plumbing between the transcript agent and the approval workflow.

The agent produces JSON. Everything downstream is deterministic Python, so a
malformed model response fails loudly here instead of quietly producing a broken
issue.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BEGIN_MARKER = "<!-- BEGIN_PROPOSALS_JSON -->"
END_MARKER = "<!-- END_PROPOSALS_JSON -->"

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


class ProposalError(ValueError):
    """Raised when agent output cannot be turned into usable proposals."""


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of possibly-decorated agent output.

    Models like to wrap JSON in markdown fences or add a sentence of preamble.
    Try the strict parse first, then a fenced block, then the outermost braces.
    """
    stripped = text.strip()
    if not stripped:
        raise ProposalError("Agent output was empty")

    for candidate in _candidates(stripped):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise ProposalError(f"Expected a JSON object at the top level, got {type(parsed).__name__}")

    preview = stripped[:400] + ("..." if len(stripped) > 400 else "")
    raise ProposalError(f"Could not find a JSON object in the agent output.\n\n{preview}")


def _candidates(text: str) -> list[str]:
    found = [text]
    found.extend(match.group(1) for match in _FENCE.finditer(text))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        found.append(text[start : end + 1])

    return found


def load_payload(path: str | Path) -> dict[str, Any]:
    """Read and parse an agent-written proposals file."""
    path = Path(path)
    if not path.exists():
        raise ProposalError(f"{path} does not exist. The agent did not produce any output.")
    return extract_json(path.read_text(encoding="utf-8"))


def validate_payload(payload: dict[str, Any], schema_path: str | Path) -> dict[str, Any]:
    """Validate against the JSON Schema and fill in optional defaults."""
    import jsonschema

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))

    if errors:
        rendered = "\n".join(
            f"  - {'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ProposalError(f"Agent output failed schema validation:\n{rendered}")

    payload.setdefault("excluded", [])
    for proposal in payload["proposals"]:
        proposal.setdefault("suggested_files", [])
        proposal.setdefault("labels", [])
        proposal.setdefault("notes", "")
        proposal.setdefault("transcript_evidence", "")

    return payload


def embed_payload(payload: dict[str, Any]) -> str:
    """Render the machine-readable block that gets appended to the issue body."""
    return "\n".join(
        [
            BEGIN_MARKER,
            "<details>",
            "<summary>Machine-readable proposal data</summary>",
            "",
            "```json",
            json.dumps(payload, indent=2, ensure_ascii=False),
            "```",
            "",
            "</details>",
            END_MARKER,
        ]
    )


def parse_embedded_payload(issue_body: str) -> dict[str, Any]:
    """Recover the payload from an issue body produced by ``embed_payload``."""
    start = issue_body.find(BEGIN_MARKER)
    end = issue_body.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ProposalError(
            "This issue has no machine-readable proposal block. It was probably not "
            "created by the transcript workflow, or the body was edited."
        )

    block = issue_body[start + len(BEGIN_MARKER) : end]
    match = _FENCE.search(block)
    if not match:
        raise ProposalError("The proposal block is present but contains no JSON code fence.")

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ProposalError(f"The embedded proposal JSON is invalid: {error}") from error

    if not isinstance(parsed, dict) or "proposals" not in parsed:
        raise ProposalError("The embedded proposal JSON is missing a 'proposals' array.")

    return parsed


def parse_selection(comment_body: str, total: int) -> list[int]:
    """Turn an ``/approve`` comment into a list of 1-based proposal numbers.

    ``/approve``       -> every proposal
    ``/approve 1,3``   -> proposals 1 and 3
    ``/approve 2-4``   -> proposals 2, 3 and 4
    """
    first_line = comment_body.strip().splitlines()[0] if comment_body.strip() else ""
    if not first_line.lower().startswith("/approve"):
        raise ProposalError(f"Not an approval command: {first_line!r}")

    argument = first_line[len("/approve") :].strip().rstrip(".")
    if not argument or argument.lower() == "all":
        return list(range(1, total + 1))

    selected: set[int] = set()
    for token in re.split(r"[,\s]+", argument):
        if not token:
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match:
            low, high = int(range_match.group(1)), int(range_match.group(2))
            if low > high:
                raise ProposalError(f"Invalid range {token!r}: start is greater than end")
            selected.update(range(low, high + 1))
            continue
        if not token.isdigit():
            raise ProposalError(
                f"Could not read {token!r} as a proposal number. "
                f"Use `/approve`, `/approve 1,3` or `/approve 2-4`."
            )
        selected.add(int(token))

    out_of_range = sorted(number for number in selected if not 1 <= number <= total)
    if out_of_range:
        raise ProposalError(
            f"Proposal number(s) {out_of_range} are out of range. This issue has {total}."
        )

    if not selected:
        raise ProposalError("No proposal numbers were given.")

    return sorted(selected)
