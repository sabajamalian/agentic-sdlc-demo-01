# Runbook

Everything you need to set the demo up once, and the script for running it live.

---

## Part 1: one-time setup

Budget 15 minutes. Steps 1 to 3 are required; the demo does not work without them.

### 1. Create the `AGENT_PAT` secret

A **fine-grained** personal access token. Classic `ghp_` tokens are not supported
by Copilot CLI and will fail with an authentication error.

Go to **Settings → Developer settings → Personal access tokens → Fine-grained tokens
→ Generate new token**.

Account permissions:

| Permission | Access |
|---|---|
| Copilot Requests | Read and write |

Repository permissions, scoped to this repository:

| Permission | Access |
|---|---|
| Metadata | Read |
| Contents | Read and write |
| Issues | Read and write |
| Pull requests | Read and write |
| Actions | Read and write |

Then store it:

```bash
gh secret set AGENT_PAT --repo <owner>/<repo>
```

Three separate things break without it, all covered in
[`architecture.md`](architecture.md#tokens): Copilot CLI cannot authenticate,
the coding agent cannot be assigned, and the workflow chain does not trigger.

### 2. Enable the Copilot cloud agent

**Settings → Copilot → Cloud agent**

- Enable the coding agent for this repository.
- Enable **"Allow GitHub Actions workflows to run without approval"**.

Skip the second one and every pull request the agent opens sits with its checks
in an approval-required state, which is a bad look mid-demo. The scheduled
sweeper in `03-pr-review-agents.yml` is a safety net for this, not a substitute.

### 3. Create the labels

```bash
scripts/bootstrap_repo.sh
```

Creates `feature-proposals`, `agent-generated`, `agent-reviewed`, `forecasting`,
`needs-human-review` and `auto-merge`, checks whether `AGENT_PAT` is present, and
reprints this checklist. Safe to re-run.

### 4. Pull request settings

**Settings → General → Pull Requests**

- Allow squash merging
- Allow auto-merge

Without these, `04-auto-merge.yml` logs a warning and does nothing.

### 5. Actions permissions

**Settings → Actions → General → Workflow permissions**

- Read and write permissions
- Allow GitHub Actions to create and approve pull requests

### 6. Optional: automatic Copilot review by ruleset

**Settings → Rules → Rulesets → New branch ruleset** targeting `main`, then enable
**"Automatically request Copilot code review"**.

The review workflow already requests it through the API, so this is belt and
braces. It does mean reviews appear a few seconds sooner during the demo.

### Verify the setup

```bash
gh secret list                                  # AGENT_PAT present
gh label list | grep feature-proposals          # labels created
gh workflow list                                # six workflows
```

---

## Part 2: the live demo

About 20 to 30 minutes of real time, most of it waiting for the coding agent.
The waits are the good part: that is where you talk about what is happening.

### Before you start

- Have the repository open on the **Actions** tab in one window and **Issues** in
  another.
- Skim [`docs/transcripts/2026-07-15-forecasting-planning.md`](transcripts/2026-07-15-forecasting-planning.md).
  Know which three items should come out and which four should not.
- If you have run the demo before, close old proposals issues so the new one is
  obvious.

### Step 1: show the starting point (2 minutes)

```bash
make check
```

Lint, the full test suite, both notebooks executed, and the eval gate. Everything green.

Then show the gaps, because these are what the agent is about to fill:

```bash
grep -rn "prophet" src/            # nothing
grep -n "on_promotion" src/        # nothing, though the column exists
head -3 data/raw/sales.csv         # on_promotion and price are right there
```

Current numbers: SARIMAX is at 9.32% MAPE against a seasonal naive baseline of
10.32%, over 5 rolling origins at a 14-day horizon. Aggregate only.

### Step 2: drop the transcript (30 seconds)

```bash
git add docs/transcripts/2026-07-15-forecasting-planning.md
git commit -m "Add forecasting planning transcript"
git push
```

Or, if the transcript is already committed:

```bash
gh workflow run 01-transcript-to-proposals.yml \
  -f transcript_path=docs/transcripts/2026-07-15-forecasting-planning.md
```

**Say while it runs (2 to 4 minutes):** the runner installs Copilot CLI, feeds it
the transcript with the prompt from `.github/agent-prompts/`, and the agent writes
`proposals.json`. It never touches the GitHub API. Deterministic Python validates
that file against a JSON Schema and renders the issue.

### Step 3: the first human gate (3 minutes)

```bash
gh issue list --label feature-proposals
```

Open the issue. Worth pointing at, in this order:

- Three proposals, each with a problem statement, acceptance criteria, likely
  files, and a quote from the transcript.
- The **"Discussed but not proposed"** table. Weather features, retraining cadence,
  the shared-runner comment. The agent filtered these and said why. This is the
  part that makes the output trustworthy.
- The collapsed machine-readable block at the bottom, which is how the next
  workflow gets structured data instead of parsing markdown.

Then approve. Approve **two of three** rather than all of them, because it makes
the point that the gate is real:

```
/approve 1,3
```

**Say while it runs (under a minute):** the workflow verifies you have write
access before doing anything, recovers the payload from the issue body, and
creates one issue per approved proposal with the Copilot coding agent as the
assignee.

### Step 4: the coding agent works (5 to 15 minutes)

```bash
gh issue list --label agent-generated
gh pr list
```

Each issue gets picked up, the agent opens a **draft** pull request on a
`copilot/` branch and shows its session log as it goes.

**Say while it runs:** the agent is reading `.github/copilot-instructions.md`,
which is why it knows to register models through `registry.py`, keep `prophet` in
an optional extra, and lag anything derived from the target. Note that it opened
a draft and cannot mark it ready itself, which is the second gate.

Good moment to open `.github/copilot-instructions.md` and show what the agent was
told.

### Step 5: the reviews (3 minutes)

On any agent pull request you should see:

1. **Copilot code review** on the diff, general code quality.
2. A **Data science review** comment from `03-pr-review-agents.yml`, looking for
   the domain-specific failures: a feature reading the target at time `t`, a
   scaler fitted before the split, a loosened threshold in `eval/thresholds.yml`,
   a model that skips the registry.
3. **CI**: lint, tests, notebook execution, and the eval gate, which posts a
   metrics table comparing the candidate to the baseline.

**The point to make:** a generic reviewer catches generic problems. Look-ahead
bias is not a code smell, it is a domain error, and it needs a reviewer that knows
what this project is. The second review is 60 lines of YAML and a prompt.

If the eval gate fails on a pull request, that is a better demo than if it passes.
Show the metrics table and the threshold it missed.

### Step 6: the second human gate (1 minute)

Read the reviews, then mark the pull request ready for review in the UI.

```bash
gh pr ready <number>
```

`04-auto-merge.yml` fires: confirms the branch is `copilot/*`, not a draft, not
labelled `needs-human-review`, and has no failing checks, then enables squash
auto-merge. GitHub merges when the last check goes green.

### Step 7: close the loop (1 minute)

```bash
gh run list --workflow=post-merge-eval.yml
```

The post-merge job runs the full backtest on `main` and uploads metrics and plots.
Pull the artifact and compare against the numbers from step 1.

Transcript to merged, evaluated model. Two human decisions in the whole chain.

---

## Part 3: when it goes wrong

### The transcript workflow fails at "Run the transcript agent"

Download the `transcript-agent-<run id>` artifact and read `agent-output.txt`.

| What you see | Cause |
|---|---|
| Authentication or 401 errors | `AGENT_PAT` is a classic token, or is missing the Copilot Requests permission |
| The agent describes proposals but writes no file | Re-run. If it repeats, tighten the last section of `.github/agent-prompts/transcript-to-proposals.md` |
| `Could not find a JSON object in the agent output` | Same, and the artifact shows exactly what came back |
| Quota or rate limit | Premium requests are exhausted for the account |

### `/approve` does nothing

Check, in order:

1. Is the issue labelled `feature-proposals`? The workflow filters on it.
2. Does the comment **start** with `/approve`? Text before it means no match.
3. Do you have write access? A read-access commenter is rejected by design.
4. Was the issue body edited? Editing away the machine-readable block breaks
   payload recovery.

The run log names whichever of these failed, and the workflow comments back on
the issue when the script errors.

### Issues are created but no agent session starts

The assignment endpoint rejects server-to-server tokens. Confirm the workflow used
`secrets.AGENT_PAT` rather than falling back to `GITHUB_TOKEN`, and that the cloud
agent is enabled for the repository.

Test the hop without touching GitHub:

```bash
python scripts/approve_proposals.py \
  --event event.json --repo <owner>/<repo> \
  --dry-run --skip-permission-check
```

### Agent pull requests sit with checks pending approval

**Settings → Copilot → Cloud agent → Allow GitHub Actions workflows to run without
approval.** Until then, release each run manually from the Actions tab.

### The review workflow never runs

The scheduled sweeper runs every 30 minutes and skips anything already labelled
`agent-reviewed`. To force it:

```bash
gh workflow run 03-pr-review-agents.yml -f pr_number=<number>
```

### Auto-merge is not enabled

Check "Allow auto-merge" and "Allow squash merging" in repository settings. Then
check the run log: the job prints exactly why it skipped a pull request (still a
draft, not a `copilot/*` branch, `needs-human-review` label, or failing checks).

---

## Cost

Every Copilot CLI invocation and every coding agent session consumes premium
requests. One full run of this demo is roughly:

| Step | Requests |
|---|---|
| Transcript parsing | 1 CLI session |
| Coding agent | 1 session per approved proposal |
| Copilot code review | 1 per pull request |
| Data science review | 1 CLI session per pull request |

Approving two proposals instead of three is a smaller bill and a better
demonstration of the gate. The scheduled review sweeper only acts on unlabelled
`copilot/*` pull requests, so it does not quietly burn quota between demos.

---

## Resetting between demos

```bash
# close leftover issues
gh issue list --label agent-generated --json number --jq '.[].number' \
  | xargs -I{} gh issue close {}
gh issue list --label feature-proposals --json number --jq '.[].number' \
  | xargs -I{} gh issue close {}

# close leftover agent pull requests and delete their branches
gh pr list --json number,headRefName \
  --jq '.[] | select(.headRefName | startswith("copilot/")) | .number' \
  | xargs -I{} gh pr close {} --delete-branch
```

Then either revert the merged agent commits on `main`, or re-run the demo from a
fresh branch.
