---
name: autoresearch-handoff
description: Use after VP passes when the experiment needs automated iteration — verifies base code, generates research protocol, and produces startup prompt for autonomous exploration
---

# Autoresearch Handoff

## Overview

Bridge between VP validation and automated research iteration. Generates `autoresearch-protocol.md` + `experiences.md` for base code that has already been built and validated.

**Core principle:** Do not rewrite VP-validated code. Handoff verifies readiness, extracts the protocol, and sets up the iteration loop.

<HARD-GATE>
Do NOT hand off without:
1. VP L1 Runtime Validation passed — mandatory for autoresearch. If skipped, STOP and run it now.
2. Design doc contains "## Autoresearch Protocol" section
</HARD-GATE>

## Checklist

1. **Verify VP L1** — record baseline metric
2. **Verify baseline speed** — first step/epoch prints fast enough
3. **Verify pressure condition termination** — time_limit / epoch_limit logic works
4. **Verify eval script is programmatic** — eval_command runs independently and produces deterministic output
5. **Profile dry-run** — if perf-mode, validate `profile_command` runs (Step 4.5)
6. **Kernel parity dry-run** — if `kernel_targets` non-empty, validate parity machinery on baseline (Step 4.6)
7. **Generate autoresearch-protocol.md** — includes Profile + Kernel Targets blocks
8. **Initialize experiences.md** — table with baseline row
9. **Verify git state** — base code committed
10. **Present launch instructions**

## Step 1: Verify VP L1

Confirm VP L1 passed. Record the evaluation metric as the protocol's baseline value.

## Step 2: Verify Baseline Speed

Check that the first step/epoch prints quickly on single GPU. This is the most important precondition for autoresearch — every round pays the cost of baseline speed. If the first step takes too long, the baseline must be fixed before entering the loop (smaller data, lighter model, faster I/O). Do NOT proceed to handoff with a slow baseline.

## Step 3: Verify Pressure Conditions

Check that time_limit and epoch_limit termination logic exists and works (VP L1 already ran under these conditions).

## Step 4: Verify Eval Script is Programmatic

**Core principle: eval must be a pre-defined, deterministic script — not agent-generated code.**

Run the eval_command independently (outside of any training context) to verify:
1. It executes without errors (given a checkpoint from VP L1)
2. It produces a parseable metric value on stdout
3. It is deterministic — running twice gives the same result
4. It is self-contained — no manual setup, no interactive prompts

If the eval script doesn't meet these criteria, fix it NOW before entering the loop. The eval script is the source of truth for the entire autoresearch loop — if it's broken or ambiguous, every round's result is unreliable.

Confirm the eval_command is listed in Fixed.files in the protocol. If not, add it.

## Step 4.5: Profile Dry-Run (skip if `profile_command` empty in design doc)

The design doc may contain a `profile_command` (collected by ml-brainstorming Q6 when `metric_category == performance`). Validate it runs against the baseline before writing the protocol:

```bash
cd <experiment_dir>
<profile_command>
```

Requirements: exit code 0; stdout non-empty (at least one kernel/op timing line).

If the design doc's `profile_command` was recorded as `TODO: build in VP L1` (user did not have one at brainstorming time), STOP the handoff and tell the user:

> "Build phase did not produce a `profile_command`. Add one to the design doc (and commit it to the experiment) before re-running handoff. We can't enter the iteration loop without a runnable profile command for perf-mode research."

If the command exists but exits non-zero, STOP and tell the user the exact error from stderr; ask them to fix `profile_command` and re-run handoff.

## Step 4.6: Kernel Parity Dry-Run (skip if `kernel_targets` empty in design doc)

For each declared kernel target:

1. **Resolve `new_module`** — check the file exists in the experiment tree. If absent, STOP and tell the user: "create `<module_path>` as a re-export of `<baseline_module>` so baseline parity is trivially passable, then re-run handoff."

2. **Run parity dry-run** — for each target in the draft protocol's `kernel_targets`, execute the inline Python heredoc from `autoresearch/SKILL.md` Step 2 (the parity check block), substituting the target's values and `$EXPERIMENT_DIR` for the placeholders.

3. Exit 0 from all targets → parity machinery is wired correctly; proceed. Exit 1 from any target → STOP and tell the user the exact `PARITY_FAIL` line from stderr. Typical causes: fixture returns wrong-shape inputs, import path typo, malformed tolerance.

This is intentionally a trivial pass — the new kernel is currently a re-export of baseline, so parity is mechanical. The dry-run's purpose is to surface **configuration** errors at handoff rather than at Round 1, where they would be misattributed to the Researcher.

## Step 5: Generate autoresearch-protocol.md

Extract from design doc's `## Autoresearch Protocol` section. Ensure eval_command's script file is included in Fixed.files. Write to `<experiment-dir>/autoresearch-protocol.md`:

````markdown
# Autoresearch Protocol: <title>

research_question: <from design doc>
max_rounds: <from design doc>
target: <from design doc, or "none">
baseline: <metric> = <value from VP L1>

## Fixed（不可变：代码 + 条件）
- files: <fixed files from design doc>
- time_limit: <value>
- epoch_limit: <value>

## Variable（可变：代码 + 条件）
- files: <variable files from design doc>
- 可调范围: <from design doc>

## Eval
- metric: <name>
- direction: <maximize / minimize>
- command: <eval_command from design doc>

<!-- Include only if design doc has profile_command -->
## Profile
- command: <profile_command from design doc>
<!-- end Profile -->

<!-- Include only if design doc has kernel_targets non-empty. The yaml fence below is intentional — Supervisor parses kernel_targets from the first ```yaml fence in the file. -->
## Kernel Targets

```yaml
kernel_targets:
  - name: <readable name>
    new_module: <module:attr>
    baseline_module: <module:attr>
    fixture: <module:attr>
    tolerance: { atol: <float>, rtol: <float> }
  # repeat per target
```
<!-- end Kernel Targets -->
````

## Step 6: Initialize experiences.md

Write to `<experiment-dir>/experiences.md`:

```markdown
# Experiences

best: {metric} = {baseline_value} (R0)
rounds: 0 / {max_rounds}
status: not_started

| Round | Compliance | Result | Verdict | Strategy | Insight | Note |
|-------|------------|--------|---------|----------|---------|------|
| 0 | ✅ | {baseline_value} | — | baseline: {brief description of baseline config} | initial | {user hints if any} |
```

## Step 7: Verify Git State

```bash
git log --oneline -1  # base code committed
```

## Step 8: Present Launch Instructions

```
Handoff complete:
- Protocol: <experiment-dir>/autoresearch-protocol.md
- Experiences: <experiment-dir>/experiences.md
- Baseline: {metric} = {value}
- Max rounds: {N}

To start — open a new session and say:

  run autoresearch at <experiment-dir>
```
