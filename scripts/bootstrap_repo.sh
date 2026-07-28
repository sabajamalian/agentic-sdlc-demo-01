#!/usr/bin/env bash
# One-time repository setup for the agentic SDLC demo.
#
# Creates the labels the workflows depend on and prints the manual checklist for
# the things the GitHub API cannot do for you.
#
# Usage:
#   scripts/bootstrap_repo.sh [owner/repo]

set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"

echo "Repository: ${REPO}"
echo

create_label() {
  local name="$1" color="$2" description="$3"
  if gh label create "${name}" --repo "${REPO}" --color "${color}" --description "${description}" 2>/dev/null; then
    echo "  created  ${name}"
  else
    gh label edit "${name}" --repo "${REPO}" --color "${color}" --description "${description}" >/dev/null
    echo "  updated  ${name}"
  fi
}

echo "Labels"
create_label "feature-proposals" "5319E7" "Transcript-derived proposals awaiting /approve"
create_label "agent-generated"   "1D76DB" "Created by the transcript-to-proposals agent"
create_label "agent-reviewed"    "0E8A16" "Reviewed by the pull request review agent"
create_label "forecasting"       "C2E0C6" "Touches the forecasting model or evaluation"
create_label "needs-human-review" "D93F0B" "Blocked on a human decision"
create_label "auto-merge"        "FBCA04" "Merge automatically once every check passes"
echo

echo "Secrets"
if gh secret list --repo "${REPO}" --json name --jq '.[].name' 2>/dev/null | grep -qx "AGENT_PAT"; then
  echo "  present  AGENT_PAT"
else
  echo "  MISSING  AGENT_PAT"
  echo "           gh secret set AGENT_PAT --repo ${REPO}"
fi
echo

cat <<'CHECKLIST'
Manual checklist (the API cannot do these for you)

1. Create a fine-grained personal access token and store it as the AGENT_PAT
   secret. Classic tokens are not supported.

   Account permissions:
     - Copilot Requests            read and write   (required by Copilot CLI)

   Repository permissions on this repo:
     - Metadata                    read
     - Contents                    read and write
     - Issues                      read and write
     - Pull requests               read and write
     - Actions                     read and write

   The default GITHUB_TOKEN cannot be used. Assigning the Copilot cloud agent
   requires a user-to-server token, and events created with GITHUB_TOKEN do not
   trigger downstream workflow runs.

2. Settings -> Copilot -> Cloud agent
     - Enable the Copilot cloud agent for this repository.
     - Enable "Allow GitHub Actions workflows to run without approval", or every
       agent pull request will sit waiting for a human to release its checks.

3. Settings -> Rules -> Rulesets  (optional but recommended)
     - New branch ruleset targeting main.
     - Enable "Automatically request Copilot code review".
     - Enable "Review new pushes" if you want a fresh review on every push.

4. Settings -> General -> Pull Requests
     - Enable "Allow auto-merge" so 04-auto-merge.yml can queue merges.
     - Enable "Allow squash merging".

5. Settings -> Actions -> General
     - Workflow permissions: read and write.
     - Allow GitHub Actions to create and approve pull requests.

Then kick off the demo:

   git add docs/transcripts/<your-transcript>.md
   git commit -m "Add planning meeting transcript"
   git push

or run the workflow manually:

   gh workflow run 01-transcript-to-proposals.yml \
     -f transcript_path=docs/transcripts/2026-07-15-forecasting-planning.md
CHECKLIST
