## What changed

<!-- One or two sentences. Lead with the outcome, not the file list. -->

Closes #

## Why

<!-- The problem this solves. Link the issue's acceptance criteria if they are not obvious. -->

## Verification

- [ ] `make lint`
- [ ] `make test`
- [ ] `make notebooks`
- [ ] `make gate`

<!-- Paste the relevant numbers if the change touches model accuracy. -->

| metric | before | after |
| --- | ---: | ---: |
|  |  |  |

## Time series checklist

- [ ] No random splits. Chronological splits only.
- [ ] Every target-derived feature is lagged by at least one step.
- [ ] Any new exogenous regressor is genuinely known at forecast time.
- [ ] New behaviour has a test that would fail without this change.

## Thresholds and dependencies

- [ ] `eval/thresholds.yml` is unchanged, **or** the change is justified below.
- [ ] No new default runtime dependency, **or** it is added as an optional extra.

<!-- If you changed a threshold or added a dependency, explain here. -->

## Left out

<!-- Anything in scope that you deliberately did not do, and why. Write "nothing" if that is the case. -->
