# ml-iteration Skill + Watchdog Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `ml-iteration` (N-round human-on-the-loop skill with Supervisor-as-Reviewer against compound criteria), narrow `watchdog` to single-run stability, extend `training-handoff` to route between them, and extract shared patterns into reference documents. `autoresearch` is not modified.

**Architecture:** Five primitive reference docs under `skills/_ml-loop-primitives/`, narrowed `watchdog/SKILL.md`, extended `ml-brainstorming/SKILL.md`, new `ml-iteration/SKILL.md`, extended `training-handoff/SKILL.md`, plus integration test and README/release-notes updates.

**Tech Stack:** SPML skill markdown, bash integration test scripts, Python (numpy-only) for test base projects.

**Spec:** `docs/superpowers/specs/2026-04-13-ml-iteration-and-watchdog-refactor-design.md`

---

## File Structure

```
skills/
├── _ml-loop-primitives/                 # NEW — reference docs (not a skill)
│   ├── researcher-dispatch.md
│   ├── scheduling-safety-net.md
│   ├── git-control.md
│   ├── experiences-log.md
│   └── eval-lock.md
├── watchdog/
│   └── SKILL.md                         # MODIFY — narrow to single-run stability
├── ml-brainstorming/
│   └── SKILL.md                         # MODIFY — add review_criteria collection
├── ml-iteration/                        # NEW
│   └── SKILL.md
├── training-handoff/
│   └── SKILL.md                         # MODIFY — route watchdog | iteration

tests/
├── ml-iteration/
│   ├── run-test.sh                      # NEW — integration flow runner
│   ├── verify.sh                        # NEW — post-run assertions
│   └── base-project/                    # NEW — minimal VP-passed project template
│       ├── train.py
│       ├── evaluate.py
│       ├── iteration-protocol.md
│       └── experiences.md

docs/
├── README.md                            # MODIFY — new capability section
└── RELEASE-NOTES.md                     # MODIFY — 0.6.0 entry

plugin.json                              # MODIFY — version bump
```

---

## Phase 1 — Shared Primitive Reference Docs

These are **documentation**, not skills. The leading `_` marks the directory as author-facing reference. Each file is small, single-responsibility, and cited by the skills in later phases.

### Task 1: Create `_ml-loop-primitives/researcher-dispatch.md`

**Files:**
- Create: `skills/_ml-loop-primitives/researcher-dispatch.md`

- [ ] **Step 1: Write the file**

```markdown
# Primitive — Researcher Subagent Dispatch

Pattern used by `ml-iteration` and `autoresearch` to hand off code modification to a subagent, then resume the Supervisor loop when the subagent returns.

## Principles

- **Fresh subagent per round.** No shared agent memory between rounds. Experience transfer happens through files (`experiences.md`, git history).
- **Supervisor injects lightweight context, not file contents.** The Researcher reads files itself — this keeps the Supervisor's context clean.
- **Background dispatch with timer pairing.** Every Researcher dispatch creates two timers (check-in + per-round timeout). Both are deleted when the Researcher completes normally.

## Prompt Skeleton

```
You are an ML researcher. Your task is to {task_description}.
This is Round {round} of {max_rounds}.

## Your role
Design a strategy and modify code. Training and evaluation are run by Supervisor
after you finish; you may smoke-test your code but do not run full training yourself.

## Constraints
- Do NOT modify any evaluation logic. Eval is a pre-defined script Supervisor owns.
- Soft boundary (ml-iteration) or Fixed/Variable (autoresearch) — {boundary_rules}.

## Recent experiences
{last_M_rounds_table_snippet}

## Task
1. Read the relevant files to understand current code.
2. Design a strategy for this round.
3. Append a row to {experiences_path} with your strategy (leave Result/Verdict blank).
4. Modify the code accordingly.
5. Report "Code ready" as your final message.
```

## Dispatch Mechanics

- Use `Agent` tool with `run_in_background: true`.
- Immediately after dispatching, create two `CronCreate` one-shots:
  - Check-in reminder (~120s) — prompts self-check whether Researcher completed.
  - Per-round timeout (`time_limit * 2`) — fails the round if Researcher hangs past budget.
- Save both job IDs. Delete both on normal completion.

## Timeout Handling

If per-round timeout fires before Researcher returns, treat the round as failed. Record diagnosis in `experiences.md` Insight, roll back, continue to next round.
```

- [ ] **Step 2: Verify the file loads as plain markdown**

Run: `head -5 skills/_ml-loop-primitives/researcher-dispatch.md`
Expected: first 5 lines of the doc above.

- [ ] **Step 3: Commit**

```bash
git add skills/_ml-loop-primitives/researcher-dispatch.md
git commit -m "docs(primitives): researcher-dispatch reference"
```

---

### Task 2: Create `_ml-loop-primitives/scheduling-safety-net.md`

**Files:**
- Create: `skills/_ml-loop-primitives/scheduling-safety-net.md`

- [ ] **Step 1: Write the file**

```markdown
# Primitive — Scheduling Safety Net

Four-layer CronCreate mechanism that prevents Supervisor loops from stalling silently when background tasks hang or the REPL goes idle.

## Layers

1. **Task-completion notification.** Dispatch long tasks with `run_in_background: true`. When the task finishes, the Supervisor is automatically notified and continues. Applies to Researcher dispatches, training runs, eval runs.

2. **Check-in reminder.** After dispatching ANY background task, immediately create a one-shot `CronCreate` to wake the Supervisor in ~120 seconds (adjust based on expected duration). When it fires:
   - If the task completed, continue.
   - If still running, create another reminder.
   - Delete the reminder when the task completes normally.

3. **Per-round timeout timer.** Before a round starts, create a one-shot `CronCreate` with `time_limit * 2`. If the round hasn't completed when it fires, mark the round failed and move on. Delete when the round completes normally.

4. **Session heartbeat.** A recurring `CronCreate` (every 30 minutes) as ultimate safety net. Fires only when REPL is idle. Self-audits: "do I have a task list? is a background task running? am I stalled?" Delete when the loop terminates.

## Hard Rule

**Never say "I'll wait" without creating a timer.** If you dispatch a background task and intend to check back later, you MUST create a `CronCreate` one-shot immediately. "I'll check in 2 minutes" without a timer is a bug — the REPL goes idle and nothing wakes you up until the task completes or the 30-minute heartbeat fires.

## Heartbeat Prompt Template

```
Supervisor heartbeat — self-audit:
- Current round has task list? If no, rebuild now.
- Writing code yourself? Stop — dispatch subagent.
- Background task running or round just finished? If neither, you're stalled — resume.
- Waiting for user confirmation? Don't — advance autonomously; user is on the loop, not in it.
Never skip steps for "simplicity". Then continue.
```

## Cleanup

On loop termination (target reached / max_rounds / user stop), delete all Supervisor-owned CronCreate jobs (by saved IDs) before the final report.
```

- [ ] **Step 2: Commit**

```bash
git add skills/_ml-loop-primitives/scheduling-safety-net.md
git commit -m "docs(primitives): scheduling-safety-net reference"
```

---

### Task 3: Create `_ml-loop-primitives/git-control.md`

**Files:**
- Create: `skills/_ml-loop-primitives/git-control.md`

- [ ] **Step 1: Write the file**

```markdown
# Primitive — Git Control

Supervisor-loop skills own all git write operations in a dedicated worktree. Researcher subagents have file-write and bash permissions but never commit, reset, or checkout.

## Rules

1. **Single writer.** Only the Supervisor performs `git commit`, `git reset`, `git checkout`, `git clean`. This guarantees a linear, auditable history.
2. **Work in a worktree.** Never run the loop on the main branch or in the main working directory. On startup:
   - Fresh: `git worktree add ../<skill>-<experiment_name> HEAD`
   - Resume: check `git worktree list`, reuse the existing worktree.
3. **.gitignore is a precondition.** Before the first round, verify `.gitignore` covers training artifacts (`outputs/`, `logs/`, `*.ckpt`, `wandb/`, etc.). If missing or incomplete, fix it before the loop starts. With a proper `.gitignore`, `git add -A` naturally skips artifacts.
4. **experiences.md survives rollback.** Before `git checkout -- . && git clean -fd`, copy `experiences.md` to `/tmp/`; restore it afterwards.

## Commit Pattern

```bash
cp experiences.md /tmp/experiences_backup.md
git add -A
git commit -m "<skill>: round {round} — <verdict_summary>"
```

## Rollback Pattern

```bash
cp experiences.md /tmp/experiences_backup.md
git checkout -- .
git clean -fd
cp /tmp/experiences_backup.md experiences.md
```

## Researcher Violation

If `git diff --name-only` after the Researcher finishes shows a locked file was modified, the round is a compliance failure. Roll back, record the violation in experiences.md Insight, continue to the next round.
```

- [ ] **Step 2: Commit**

```bash
git add skills/_ml-loop-primitives/git-control.md
git commit -m "docs(primitives): git-control reference"
```

---

### Task 4: Create `_ml-loop-primitives/experiences-log.md`

**Files:**
- Create: `skills/_ml-loop-primitives/experiences-log.md`

- [ ] **Step 1: Write the file**

```markdown
# Primitive — Experiences Log

`experiences.md` is the shared memory of a Supervisor loop across rounds. Append-only. Two variants share the same header format and differ in per-round columns.

## Header

```markdown
# Experiences — <experiment_name>

- mode: iteration | autoresearch
- best_commit: <hash> (round {N})
- best_state:
    {metric_or_criteria_snapshot}
- rounds: {completed_round_count}
- status: not_started | running | completed | target_reached | stopped_by_user

---
```

The Supervisor updates `best_commit`, `best_state`, `rounds`, and `status` after each round.

## Row (autoresearch variant)

| Round | Strategy | {metric} | Verdict | Insight |
|-------|----------|----------|---------|---------|
| 1 | cosine lr + label smoothing | 0.813 | committed | label smoothing improved over warmup |
| 2 | mixup aug | 0.799 | rolled_back | mixup hurt early-epoch accuracy |

## Row (iteration variant)

| Round | Strategy | Metrics | Speed | Observability | Stability | User Hint | Verdict | Insight |
|-------|----------|---------|-------|---------------|-----------|-----------|---------|---------|
| 1 | ... | ... | ... | ... | ... | "next focus: log format" | committed | ... |

Each column corresponds to a `review_criteria` dimension (plus a `User Hint` column for human-on-the-loop input). Dimensions absent from the design doc's review_criteria get omitted columns.

## Discipline

- **Append-only within a round.** Once a verdict is recorded, the row is not rewritten. Supervisor-override of a prior verdict creates a new row describing the override.
- **Researcher writes the Strategy column only.** Everything else (metrics, verdict, insight) is written by the Supervisor.
- **Insight explains cause, not outcome.** For rolled-back rounds, the Insight must help the next round's Researcher avoid the same failure.
```

- [ ] **Step 2: Commit**

```bash
git add skills/_ml-loop-primitives/experiences-log.md
git commit -m "docs(primitives): experiences-log reference"
```

---

### Task 5: Create `_ml-loop-primitives/eval-lock.md`

**Files:**
- Create: `skills/_ml-loop-primitives/eval-lock.md`

- [ ] **Step 1: Write the file**

```markdown
# Primitive — Eval Lock

The evaluation script is a pre-defined, deterministic program. Neither the Researcher nor the Supervisor may modify its logic during the loop.

## Why

If the Researcher could write or change eval code, it could (intentionally or not) produce favorable metrics that mislead the loop. Locking eval is the single largest protection against agent self-deception in autonomous ML iteration.

## Mechanics

- Eval script path is fixed at protocol-generation time (during `training-handoff` or `autoresearch-handoff`) and recorded in the protocol file.
- The script lives in **locked files** (iteration) or **Fixed.files** (autoresearch). Any modification is a compliance violation.
- **Supervisor never substitutes training-log metrics for eval.** The eval command is the only source of truth. If it fails, fix the environment, paths, or missing deps — never the eval logic.
- If the Researcher creates any new eval-like script (even in new files that are not nominally locked), the Supervisor ignores those and uses the original `eval_command` only.

## Enforcement Hook

In the Supervisor's compliance check, beyond the `git diff --name-only` check for locked files, also grep for common eval-function names (`evaluate`, `compute_metrics`, `accuracy`, `score`) in any newly created files. Flag matches for user review.

## When the Eval Needs to Change

If the eval script itself is buggy or must evolve, the loop must be stopped. Changes to eval logic belong in a new experiment or a new handoff cycle, not in the middle of a running loop. This is not a limitation — it is the guarantee that every round's metric is comparable.
```

- [ ] **Step 2: Commit**

```bash
git add skills/_ml-loop-primitives/eval-lock.md
git commit -m "docs(primitives): eval-lock reference"
```

---

## Phase 2 — Narrow `watchdog`

Remove Tier 2 (parameter auto-fix), Tier 3 (code-fix sub-agent), and the three-mode system. Keep Tier 1 (env restart), async evaluation, and baseline-deviation alerts.

### Task 6: Narrow `watchdog/SKILL.md`

**Files:**
- Modify: `skills/watchdog/SKILL.md`

- [ ] **Step 1: Read the current file to locate the sections to remove**

Run: `wc -l skills/watchdog/SKILL.md && grep -n '^##\|^###' skills/watchdog/SKILL.md`
Expected output: section index showing current structure (Overview, Operating Modes, Startup, Problem Classification, Tier 1/2/3, Monitoring Loop, Restart, Async Eval, ...).

- [ ] **Step 2: Replace the "Operating Modes" section — collapse to a single mode**

Old content to find: the `## Operating Modes` section with the three-mode table.

New content:

```markdown
## Scope

Watchdog keeps a single training run healthy. Its only intervention is restarting from the latest checkpoint after an environment failure. It does not change parameters, fix code, or decide what to do next — those belong in `ml-iteration` or `autoresearch`.

If you need iterative parameter or code changes, stop watchdog and re-run `training-handoff` to pick `ml-iteration` instead.
```

- [ ] **Step 3: Delete Tier 2 and Tier 3 sections entirely**

Find and remove:
- `### Tier 2: Simple Parameter Problems` and its body.
- `### Tier 3: Complex Problems` and its body.
- Any reference to "Guardian" / "Autonomous" / "Monitor" modes in startup, monitoring-loop, and anomaly sections.

Keep:
- `### Tier 1: Environment Problems` — unchanged, with restart-from-checkpoint behavior.

- [ ] **Step 4: Simplify "Problem Classification" intro**

Replace multi-tier framing with:

```markdown
## Problem Classification

Two outcomes only:

- **Environment problem** (OOM killer, NCCL timeout, hardware error, disk full, SIGKILL, hang past baseline × 10) → restart from latest checkpoint. No retry limit; if crashes persist (e.g., 5+ within 30 minutes), surface a warning but keep retrying.
- **Anything else** (code bug, wrong metric trend, NaN in inputs, plateau past VP baseline) → report to the user, do not auto-fix. Write a diagnosis to `experiment-context.md` and notify the user.
```

- [ ] **Step 5: Add a cross-reference to primitives**

Just below the `## Overview` section, add:

```markdown
## Shared Patterns

This skill uses the following primitive patterns — see `skills/_ml-loop-primitives/` for details:
- `scheduling-safety-net.md` — monitoring loop timer discipline.
```

(Watchdog does not use git-control, researcher-dispatch, experiences-log, or eval-lock.)

- [ ] **Step 6: Update the frontmatter description to reflect the narrowed scope**

Old:
```
description: Use when monitoring a long-running ML task — active shepherd with three operating modes, automatic restart, parameter fixing, and sub-agent spawning for complex issues
```

New:
```
description: Use when monitoring a single long-running ML training run — restarts from checkpoint on environment failures, runs async evaluation on new checkpoints, and surfaces anomalies for the user to handle
```

- [ ] **Step 7: Update the `## When to Use` section**

Replace with:

```markdown
## When to Use

- User has pasted a Watchdog prompt from `training-handoff` (watchdog branch).
- A single long-running training run needs supervision, and the user does not want N-round iteration.

## When Not to Use

- User wants parameter tuning, code fixes, or multiple training rounds → use `ml-iteration`.
- User wants metric-driven search with Fixed/Variable file partitions → use `autoresearch`.
```

- [ ] **Step 8: Verify the file no longer mentions deprecated concepts**

Run: `grep -nE "Guardian|Autonomous|Monitor mode|Tier 2|Tier 3|three-mode|watchdog_mode" skills/watchdog/SKILL.md`
Expected: no matches. If any remain, they were missed in earlier steps — remove.

- [ ] **Step 9: Commit**

```bash
git add skills/watchdog/SKILL.md
git commit -m "refactor(watchdog): narrow scope to single-run stability only"
```

---

## Phase 3 — Extend `ml-brainstorming`

Collect `review_criteria` as a required dimension of experiment design, regardless of which post-handoff path the user picks.

### Task 7: Add `review_criteria` collection to `ml-brainstorming/SKILL.md`

**Files:**
- Modify: `skills/ml-brainstorming/SKILL.md`

- [ ] **Step 1: Locate the Eval design section**

Run: `grep -n '^##\|^###' skills/ml-brainstorming/SKILL.md`
Expected: section index. Look for the section that discusses evaluation design (it comes after metric definition in the current flow).

- [ ] **Step 2: Insert the `review_criteria` collection block immediately after the Eval design section**

Content to insert:

```markdown
### Review Criteria (compound)

After defining the eval metric, collect the full set of acceptance dimensions for this experiment. These are used by `verification` to judge overall success, and by `ml-iteration` as the Supervisor's review rubric.

Ask:

> "Beyond the metric, what does a 'good-enough' version of this training look like? Consider speed (first-step time, MFU), log quality (what fields, what cadence), stability (no NaN, no crashes), and anything else we discussed. I'll record these as review_criteria."

Record the response as a structured block in the design doc:

```yaml
review_criteria:
  metrics:
    - name: <eval metric name>
      direction: ">=" | "<=" | "=="
      threshold: <value>
  performance:
    - <constraint, e.g., "first_step_time <= 30s">
    - <constraint, e.g., "mfu >= 0.30">
  observability:
    - <expectation, e.g., "per-step loss / grad_norm / step_time">
  stability:
    - <expectation, e.g., "no NaN / Inf">
  custom:
    - <any other expectation raised in the conversation>
```

Any sub-section may be empty if not discussed. `metrics` is strongly recommended; others are context-dependent.

**This field is required in the design doc** — `training-handoff` will prompt the user to add it if missing.
```

- [ ] **Step 3: Add a cross-reference in the output-checklist portion of the skill**

Find the section that enumerates what the design doc must contain (usually a checklist). Add:

```markdown
- [ ] `review_criteria` block with at least `metrics` populated
```

- [ ] **Step 4: Verify**

Run: `grep -n 'review_criteria' skills/ml-brainstorming/SKILL.md`
Expected: at least 3 matches (section, yaml block, checklist).

- [ ] **Step 5: Commit**

```bash
git add skills/ml-brainstorming/SKILL.md
git commit -m "feat(ml-brainstorming): collect review_criteria in design doc"
```

---

## Phase 4 — New `ml-iteration` Skill

### Task 8: Create `ml-iteration/SKILL.md` — frontmatter, overview, hard gates

**Files:**
- Create: `skills/ml-iteration/SKILL.md`

- [ ] **Step 1: Write the file header**

```markdown
---
name: ml-iteration
description: Use when running an N-round iterative training loop — Supervisor dispatches Researcher subagents, runs training + fixed eval, reviews against compound criteria, commits improvements, and keeps the human on the loop for supplementary feedback
---

# ML Iteration

## Overview

N-round iteration Supervisor for post-handoff training that is not ready to ship. Each round dispatches a fresh Researcher to modify code, then the Supervisor runs training + eval, produces a compound review against `review_criteria`, and commits or rolls back. The human is on the loop: they watch, override, and re-aim rounds — they do not gate each round.

**Core principle (differs from `autoresearch`):** review is compound and LLM-judged. There is no single metric and no Pareto rule. The Supervisor acts as Reviewer against the design-doc's `review_criteria`, producing a verdict that the human can override.

**Core principle (shared with `autoresearch`):** the Supervisor owns git writes, the eval script is locked, and each round gets a fresh Researcher.

## Shared Patterns

This skill uses the following primitive patterns — see `skills/_ml-loop-primitives/` for details:
- `researcher-dispatch.md`
- `scheduling-safety-net.md`
- `git-control.md`
- `experiences-log.md`
- `eval-lock.md`

<HARD-GATE>
## Git Control

You MUST be the ONLY entity that performs git write operations (commit, checkout, reset).
Researcher has file read/write and bash permissions but NO git write permissions.
All git operations happen in the worktree — NEVER in the main working directory.
</HARD-GATE>

<HARD-GATE>
## Monitoring Loop Mechanism

You MUST use a subagent (Agent tool) to dispatch Researcher.
Each round is a fresh subagent — no shared context between rounds.
Experience transfer happens through files (`experiences.md`, git history), not agent memory.

Four-layer scheduling — see `scheduling-safety-net.md` primitive:
1. Researcher notification (background dispatch)
2. Check-in reminder (~120s after dispatch)
3. Per-round timeout (`time_limit * 2`)
4. Session heartbeat (every 30 min)

**Never say "I'll wait" without a timer.**
</HARD-GATE>
```

- [ ] **Step 2: Commit the skeleton**

```bash
git add skills/ml-iteration/SKILL.md
git commit -m "feat(ml-iteration): skill skeleton with frontmatter and hard gates"
```

---

### Task 9: Add `ml-iteration/SKILL.md` — Startup + Main Loop

**Files:**
- Modify: `skills/ml-iteration/SKILL.md`

- [ ] **Step 1: Append Startup section**

```markdown
## When to Use

- User has pasted an iteration startup prompt from `training-handoff` (iteration branch).
- An experiment has passed VP but needs multi-round code iteration against compound review criteria.

## Startup

1. **Read `iteration-protocol.md`** — extract: max_rounds, time_limit, train_command, eval_command, review_criteria, focused_files, locked_files, initial_hints.
2. **Create or reuse worktree** (see `git-control.md`): `git worktree add ../ml-iteration-{experiment_name} HEAD` for fresh, reuse existing if present.
3. **Verify .gitignore** covers training artifacts (outputs, logs, checkpoints, wandb). Fix if incomplete.
4. **Check for resume:**
   - Read `experiences.md` header → extract `rounds` and `status`.
   - `status: running` + rounds > 0 → resuming, round = rounds + 1.
   - `status: not_started` → fresh start, round = 1.
   - `status: completed` → ask the user whether to extend rounds or finish.
5. **Announce:** "Starting ml-iteration on {experiment}. Round {current}/{max_rounds}. Criteria: {summary}."
6. **Set up session heartbeat** (see `scheduling-safety-net.md`), save job ID.
7. **Enter main loop.**
```

- [ ] **Step 2: Append Main Loop section**

```markdown
<HARD-GATE>
## Main Loop

The loop is autonomous. Never stop unless the user explicitly says to stop, or termination conditions are met.

**User input during the loop** — handle inline:
- **Stop command** ("stop", "pause") → finish current step, terminate.
- **Protocol change** ("also watch X", "increase max_rounds") → update `iteration-protocol.md` directly, continue.
- **Next-round hint** ("next round focus on log format") → append to current round's User Hint field; injected into next Researcher prompt.
- **Verdict override** ("roll back that last commit") → execute the git operation, record the override in experiences.
- **Question** → answer briefly, continue.

Never stop for anything other than explicit stop or termination.
</HARD-GATE>

```
for round in current_round..max_rounds:

  0. CREATE ROUND TASK LIST       (S1–S6)
  1. DISPATCH RESEARCHER
  2. COMPLIANCE CHECK             (locked_files untouched?)
  3. TRAIN                        (Supervisor runs train_command)
  4. EVALUATE                     (Supervisor runs eval_command — locked)
  5. COMPOUND REVIEW + VERDICT    (pass/fail/improved/regressed per criterion → commit | rollback)
  6. ACT ON VERDICT               (git commit or rollback, preserving experiences.md)
  7. ABSORB USER INPUT            (hints / overrides received since last round)
  8. CHECK TERMINATION            (all criteria met / max_rounds / user stop)
  9. REPORT PROGRESS
```
```

- [ ] **Step 3: Commit**

```bash
git add skills/ml-iteration/SKILL.md
git commit -m "feat(ml-iteration): startup and main-loop skeleton"
```

---

### Task 10: Add `ml-iteration/SKILL.md` — Step details 0–4

**Files:**
- Modify: `skills/ml-iteration/SKILL.md`

- [ ] **Step 1: Append Step 0 — Task List**

```markdown
<HARD-GATE>
### Step 0: Create Round Task List

Create the task list BEFORE dispatching Researcher, for EVERY round.

**Self-check (anti-laziness):** if mid-loop you notice you dispatched Researcher without an S1–S6 list, that is a protocol violation. Stop, build the list now, record the violation in `experiences.md` Insight, continue.

Clear previous round's tasks, then create:

```
TaskCreate: "R{round} S1: Researcher — {task_description from user hint or default}"
TaskCreate: "R{round} S2: Compliance — verify locked_files untouched"
TaskCreate: "R{round} S3: Train — {train_command} (limit: {time_limit})"
TaskCreate: "R{round} S4: Eval — {eval_command}"
TaskCreate: "R{round} S5: Review — compound verdict vs review_criteria"
TaskCreate: "R{round} S6: Termination — {round}/{max_rounds}"
```

Update with actual results on completion (strategy summary, verdict, metric snapshot).
</HARD-GATE>
```

- [ ] **Step 2: Append Step 1 — Dispatch Researcher**

```markdown
### Step 1: Dispatch Researcher

Assemble the prompt per `researcher-dispatch.md`, injecting:

- `review_criteria` (full block).
- Current best-commit's state across each dimension (from `experiences.md` header).
- Last M rounds of experiences table (include rolled-back rounds as negative examples; default M = 5).
- Latest user hint (from the User Hint column of the current round's placeholder row, or empty).
- `focused_files`, `locked_files`, and "other files are soft — you may modify them but doing so will be recorded."

Prompt body:

```
You are an ML researcher. Your task is to improve this training against the
compound review_criteria below. This is Round {round} of {max_rounds}.

## Your role
Design a strategy and modify code. Supervisor runs training and evaluation.

## Review criteria
{review_criteria full yaml}

## Current best state
{best_commit state per criterion}

## Recent experiences (last {M} rounds)
{experiences table snippet}

## User hint for this round
{user_hint or "none"}

## Boundaries
- Locked files (do NOT modify): {locked_files}
- Focused files (primary target): {focused_files}
- Other files: soft boundary — modifying them is allowed but will be recorded.
- Do NOT create or modify any evaluation logic. Eval is owned by Supervisor.

## Task
1. Read relevant files.
2. Design a strategy for this round.
3. Append a row to {experiences_path} with Strategy filled; leave Verdict and Insight blank.
4. Modify code accordingly. You may run quick smoke tests.
5. Report "Code ready" as your final message.
```

Dispatch with `run_in_background: true`. Create two timers (check-in + per-round timeout). Save their IDs.

When the subagent returns, delete both timers and proceed to Step 2.
```

- [ ] **Step 3: Append Steps 2, 3, 4**

```markdown
### Step 2: Compliance Check

```bash
git diff --name-only HEAD
```

Check that no `locked_files` were modified. If any was → this round is a violation: roll back, record in `experiences.md` Insight ("modified locked file {path}"), skip to Step 7.

Also grep newly-created files for eval-like function names (`evaluate`, `compute_metrics`, `accuracy`, `score`) and flag matches for user review; this is advisory, not a hard block.

### Step 3: Train

```
Bash(
  command: "{train_command}",
  run_in_background: true,
  timeout: {time_limit_ms + 30000}
)
```

Immediately create a check-in reminder at ~80% of `time_limit`. Save the ID, delete on completion.

If Bash timeout fires → round fails with "training exceeded time_limit"; insight in experiences. Skip to Step 7 with rollback verdict.

### Step 4: Evaluate

<HARD-GATE>
Run `eval_command` EXACTLY as recorded in `iteration-protocol.md`. Do not wrap, modify, or substitute with training-log metrics. If the command fails:

1. Check if the failure is environment (missing dep, wrong path) — fix those.
2. Do NOT modify eval logic, even in new files.
3. If unfixable → pause the loop, notify the user, wait for instruction.
</HARD-GATE>

Parse the metric outputs. Collect each dimension's observable value (metrics from eval; speed/observability/stability from training log; custom from whatever the criterion says). Pass the full snapshot to Step 5.
```

- [ ] **Step 4: Commit**

```bash
git add skills/ml-iteration/SKILL.md
git commit -m "feat(ml-iteration): steps 0-4 detail"
```

---

### Task 11: Add `ml-iteration/SKILL.md` — Steps 5–9, anomaly, termination

**Files:**
- Modify: `skills/ml-iteration/SKILL.md`

- [ ] **Step 1: Append Step 5 — Compound Review**

```markdown
### Step 5: Compound Review and Verdict

Walk each criterion against current best-commit's state:

```
for criterion in review_criteria:
    current_value = observed this round
    best_value    = state of current best commit
    threshold     = criterion.threshold (if defined)

    status =
        "pass"       if criterion.threshold met
        "improved"   if current_value is better than best_value and threshold not yet met
        "regressed"  if current_value is worse than best_value (beyond noise)
        "fail"       if no threshold, just missing
```

Emit the verdict:

| Condition | Verdict |
|-----------|---------|
| All criteria pass | `commit` + terminate (target reached) |
| No regression, at least one improvement | `commit` |
| Any clear regression | `rollback` |
| Ambiguous (small noise, mixed signal) | `accept_with_note` (commit, flagged) |

Record the verdict, the per-criterion status map, and a short rationale in `experiences.md`.

**LLM judgment, not rigid rule.** "Clearly regressed" is a judgment — small noise-scale movements are not regressions. When unsure, prefer `commit` and flag; the human can override.
```

- [ ] **Step 2: Append Step 6 — Act on Verdict**

```markdown
### Step 6: Act on Verdict

**If commit or accept_with_note:**
```bash
cp experiences.md /tmp/experiences_backup.md
git add -A
git commit -m "ml-iteration: round {round} — {verdict summary}"
```
Update `experiences.md` header: best_commit, best_state (if improved), rounds += 1.

**If rollback:**
```bash
cp experiences.md /tmp/experiences_backup.md
git checkout -- .
git clean -fd
cp /tmp/experiences_backup.md experiences.md
```
Update `experiences.md`: fill in the current row's Verdict (`rolled_back`) and Insight (explain the regression so the next round's Researcher knows).
```

- [ ] **Step 3: Append Steps 7, 8, 9**

```markdown
### Step 7: Absorb User Input

Check any user messages received during this round. Apply per the Main Loop's intent-routing table. Hints update the next round's prompt; overrides update git + experiences.

### Step 8: Check Termination

Terminate ONLY for:
- All `review_criteria` pass → `target_reached`
- `round == max_rounds` → `completed`
- User stop command → `stopped_by_user`

Otherwise continue to the next round. Do NOT stop on a judgment call like "the metric can't improve further."

### Step 9: Report Progress

```
Round {round}/{max_rounds}: verdict={verdict}
  Criteria status: metrics={p/f/i/r}, performance={...}, observability={...}, stability={...}, custom={...}
  Best so far: commit={hash} (round {N})
```
```

- [ ] **Step 4: Append Anomaly Handling**

```markdown
## Anomaly Handling

| Anomaly | Action |
|---------|--------|
| Researcher timeout / crash | Round fails; rollback; insight records cause; no auto-retry |
| Training env crash (OOM, NCCL) | Round fails; if persistent, stop and re-handoff into `watchdog` mode |
| Training exceeds time_limit | Backstop fires; treat as env crash |
| eval_command failure | Fix environment/paths only; never modify eval logic; unfixable → pause + notify |
| Locked files modified | rollback + experiences flag; does not count as a valid round |
| N consecutive rollbacks | Plateau warning to user; continue (not a termination condition) |

### Session Interruption Recovery

On startup, if `experiences.md` shows `status: running` with rounds > 0:
1. Verify git HEAD matches latest committed improvement.
2. Last row has no verdict → mid-round interruption; rollback, restart that round.
3. Last row has verdict → continue from next round.
```

- [ ] **Step 5: Append Final Report section**

```markdown
## Final Report

On termination:
1. Delete heartbeat cron (by saved ID).
2. Update `experiences.md` status.
3. Present worktree options: merge / keep / remove.

```
# ml-iteration Complete

## Result
- Status: {target_reached | completed | stopped_by_user}
- Rounds: {completed} / {max_rounds}
- Best commit: {hash} (round {N})
- Criteria status: {summary per dimension}

## Key Insights
<Distill top insights from experiences.md>
```
```

- [ ] **Step 6: Commit**

```bash
git add skills/ml-iteration/SKILL.md
git commit -m "feat(ml-iteration): steps 5-9, anomaly, termination, final report"
```

---

## Phase 5 — Extend `training-handoff`

### Task 12: Add watchdog/iteration routing to `training-handoff/SKILL.md`

**Files:**
- Modify: `skills/training-handoff/SKILL.md`

- [ ] **Step 1: Read current structure**

Run: `grep -n '^##\|^###' skills/training-handoff/SKILL.md`
Expected: current section index.

- [ ] **Step 2: Insert a "Routing" section right after Overview**

```markdown
## Routing

After VP validation passes, ask the user to choose between two post-handoff paths:

> "This experiment is validated. Two ways to run from here:
>
> 1. **watchdog** — run training once, auto-restart on environment failures, async eval on new checkpoints. Choose this when the training script is ready to ship and you just want it supervised.
>
> 2. **ml-iteration** — run N short rounds (~10 min each). Each round a Researcher subagent modifies code, Supervisor trains + evaluates, Supervisor reviews against `review_criteria`, you can interject any time. Choose this when the training script still has rough edges (speed, log format, stability, partially satisfied metrics).
>
> Which?"

**Default suggestion:**
- `review_criteria` has at least one unmet dimension in VP results → suggest `ml-iteration`.
- All `review_criteria` already passed in VP L1 → suggest `watchdog`.
- Brainstorming explicitly flagged single-run intent → suggest `watchdog`.

The choice branches the remainder of the handoff into one of the two sub-flows below.
```

- [ ] **Step 3: Add the `review_criteria` gate**

Insert just before the Routing section:

```markdown
## Preconditions

1. All enabled VP layers passed, baseline numbers recorded.
2. Design doc contains `review_criteria` block — at minimum the `metrics` sub-section. If missing, halt handoff and ask the user to add it to the design doc (this is the compass for both paths).
```

- [ ] **Step 4: Refactor the existing checklist into "Watchdog branch" sub-section**

Wrap the current checklist (verify VP, verify directory, verify training script, write experiment-context.md, write watchdog-prompt.md, present launch instructions) under:

```markdown
## Watchdog Branch

(invoked when user picks watchdog)

<the existing checklist as-is>
```

- [ ] **Step 5: Add "Iteration branch" sub-section**

```markdown
## Iteration Branch

(invoked when user picks ml-iteration)

1. **Verify training script readiness** — same "Required" checks as watchdog branch (standalone runnable, log to file, fixed seeds).
2. **Verify directory layout** — same as watchdog branch (co-located artifacts).
3. **Extract iteration parameters** — from the design doc and user:
   - `max_rounds` — default 10, ask to override.
   - `time_limit` — per-round training budget. Default: `time_limit` used in VP L1 (typically 5–10 min).
   - `focused_files` — "which files do you expect to change most this iteration?" (soft boundary hint, can be empty).
   - `locked_files` — auto-populated: `eval_command` script path + core data loader path.
   - `initial_hints` — optional first-round Researcher hint.
4. **Generate `iteration-protocol.md`** at the experiment directory — template below.
5. **Generate `experiences.md`** with initial header (mode: iteration, status: not_started, rounds: 0).
6. **Generate `iteration-prompt.md`** — a startup prompt the user pastes into a new session to launch `ml-iteration`.
7. **Present launch instructions** — "Start a new Claude session in this directory, paste iteration-prompt.md."

### iteration-protocol.md Template

```yaml
---
mode: iteration
experiment: {experiment_name}
max_rounds: {max_rounds}
time_limit: {time_limit}
train_command: {train_command from VP L1}
eval_command: {eval_command from VP L1}
---

## Review criteria
{copied verbatim from design doc}

## Modification boundary (soft)
- Focused_files: {focused_files}
- Locked_files: {locked_files}
- Other: soft — Researcher may modify; will be recorded in experiences.md

## Initial hints
{initial_hints or "(none)"}
```

### iteration-prompt.md Template

```markdown
Run ml-iteration at {experiment_dir}.

Protocol: {experiment_dir}/iteration-protocol.md
Experiences log: {experiment_dir}/experiences.md

Use the `spml:ml-iteration` skill.
```
```

- [ ] **Step 6: Update frontmatter description**

Old:
```
description: Use after VP passes when the task includes a long-running phase — verifies training script readiness, writes experiment context file, and generates Watchdog prompt for monitoring
```

New:
```
description: Use after VP passes when the task needs post-validation supervision — routes between watchdog (single run) and ml-iteration (N rounds), generating the appropriate protocol and startup prompt
```

- [ ] **Step 7: Commit**

```bash
git add skills/training-handoff/SKILL.md
git commit -m "feat(training-handoff): route between watchdog and ml-iteration"
```

---

## Phase 6 — Integration Test

A minimal end-to-end exercise that validates the flow, not the ML correctness.

### Task 13: Create `tests/ml-iteration/` harness

**Files:**
- Create: `tests/ml-iteration/base-project/train.py`
- Create: `tests/ml-iteration/base-project/evaluate.py`
- Create: `tests/ml-iteration/base-project/iteration-protocol.md`
- Create: `tests/ml-iteration/base-project/experiences.md`
- Create: `tests/ml-iteration/run-test.sh`
- Create: `tests/ml-iteration/verify.sh`

- [ ] **Step 1: Create `base-project/train.py` — minimal fake training**

```python
import json
import os
import random
import sys
import time

STEPS = int(os.environ.get("STEPS", 20))
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)

out_dir = os.environ.get("OUT_DIR", "outputs")
os.makedirs(out_dir, exist_ok=True)

for step in range(STEPS):
    loss = 1.0 / (1 + step * 0.1) + random.uniform(-0.01, 0.01)
    print(f"step={step} loss={loss:.4f} step_time=0.01s", flush=True)
    time.sleep(0.01)

# Write fake checkpoint
with open(os.path.join(out_dir, "ckpt.json"), "w") as f:
    json.dump({"final_loss": loss, "steps": STEPS, "seed": SEED}, f)

print("Training done.", flush=True)
```

- [ ] **Step 2: Create `base-project/evaluate.py` — deterministic score**

```python
import json
import os
import sys

ckpt_path = os.environ.get("CKPT", "outputs/ckpt.json")

with open(ckpt_path) as f:
    ckpt = json.load(f)

# Deterministic "accuracy" from final_loss — monotonic: lower loss → higher accuracy
accuracy = max(0.0, 1.0 - ckpt["final_loss"])
print(f"accuracy={accuracy:.4f}")
print(f"duration_s=0.2")
```

- [ ] **Step 3: Create `base-project/iteration-protocol.md`**

```markdown
---
mode: iteration
experiment: test-exp
max_rounds: 3
time_limit: 30s
train_command: STEPS=20 python train.py
eval_command: python evaluate.py
---

## Review criteria
metrics:
  - name: accuracy
    direction: ">="
    threshold: 0.95
performance:
  - first_step_time: "<= 1s"
observability:
  - "per-step loss output"
stability:
  - "no NaN"

## Modification boundary (soft)
- Focused_files: [train.py]
- Locked_files: [evaluate.py]
- Other: soft

## Initial hints
Increase STEPS to reach accuracy threshold.
```

- [ ] **Step 4: Create `base-project/experiences.md`**

```markdown
# Experiences — test-exp

- mode: iteration
- best_commit: (none)
- best_state: (none)
- rounds: 0
- status: not_started

---

| Round | Strategy | Metrics | Speed | Observability | Stability | User Hint | Verdict | Insight |
|-------|----------|---------|-------|---------------|-----------|-----------|---------|---------|
```

- [ ] **Step 5: Create `tests/ml-iteration/run-test.sh`**

```bash
#!/usr/bin/env bash
# Dry run: just verify the protocol/experiences files parse and the scripts execute.
set -euo pipefail

cd "$(dirname "$0")/base-project"
export OUT_DIR="$(mktemp -d)"

# Exercise train + eval as Supervisor would
STEPS=20 python train.py
python evaluate.py > "$OUT_DIR/eval_out.txt"

cat "$OUT_DIR/eval_out.txt"

# Bump STEPS to demonstrate the "Researcher modifies script" behavior
STEPS=200 python train.py
python evaluate.py > "$OUT_DIR/eval_out_2.txt"

cat "$OUT_DIR/eval_out_2.txt"
```

- [ ] **Step 6: Create `tests/ml-iteration/verify.sh`**

```bash
#!/usr/bin/env bash
# Verify that accuracy monotonically improves as STEPS grows
# (sanity check for the test harness, not for the skill itself).
set -euo pipefail

cd "$(dirname "$0")/base-project"

STEPS=20  python train.py > /dev/null
acc1=$(python evaluate.py | awk -F= '/accuracy/{print $2}')

STEPS=200 python train.py > /dev/null
acc2=$(python evaluate.py | awk -F= '/accuracy/{print $2}')

python3 -c "
import sys
a1, a2 = float('$acc1'), float('$acc2')
print(f'acc @ STEPS=20: {a1}')
print(f'acc @ STEPS=200: {a2}')
sys.exit(0 if a2 > a1 else 1)
"
```

- [ ] **Step 7: Make scripts executable, run them**

```bash
chmod +x tests/ml-iteration/run-test.sh tests/ml-iteration/verify.sh
tests/ml-iteration/run-test.sh
tests/ml-iteration/verify.sh
```

Expected: `verify.sh` exits 0 with acc @ STEPS=200 > acc @ STEPS=20. This validates the test harness — not the skill.

- [ ] **Step 8: Commit**

```bash
git add tests/ml-iteration/
git commit -m "test(ml-iteration): base-project harness and verify script"
```

---

## Phase 7 — README + Release Notes + Version Bump

### Task 14: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the tagline paragraph at the top**

Old (first paragraph after `# SPML — ML SuperPowers`):
```
SPML is an addon plugin for [Superpowers](https://github.com/obra/superpowers) that extends it with ML experiment workflows: Validation Pyramid, experiment-driven development, Watchdog-based training monitoring, and Auto Research — an autonomous iteration loop where a Supervisor dispatches Researcher subagents, runs evaluation, manages git, and accumulates experience across rounds.
```

New:
```
SPML is an addon plugin for [Superpowers](https://github.com/obra/superpowers) that extends it with ML experiment workflows: Validation Pyramid, experiment-driven development, Watchdog-based single-run supervision, ml-iteration (N-round human-on-the-loop iteration against compound criteria), and Auto Research (protocol-driven metric search).
```

- [ ] **Step 2: Update the "SPML addresses this with" bullet list**

Replace the current list with:

```markdown
SPML addresses this with:
- **Validation Pyramid** — 2-level verification (static analysis, runtime + pipeline validation) that separates "implementation bug" from "strategy doesn't work"
- **Watchdog** — single-run training supervision: restarts from checkpoint on environment failures, async evaluation on new checkpoints, baseline-deviation alerts
- **ml-iteration** — N-round Supervisor-driven iteration against compound review criteria: Researcher subagents modify code each round, Supervisor reviews and commits or rolls back, human on the loop can interject
- **Auto Research** — protocol-driven autonomous iteration: Supervisor dispatches fresh Researcher subagents each round, runs the fixed eval script, commits improvements and rolls back regressions, and passes lessons between rounds through an experiences log
- **Experiment-driven workflow** — hypothesis, independent/dependent/control variables, conclusion recording with metric evidence
```

- [ ] **Step 3: Update the Workflow diagram**

Replace the current workflow block in "The ML Workflow" with:

```
ml-brainstorming
    Refine hypothesis, collect context, define review_criteria (compound)
    |
experiment-planning
    Break into atomic subtasks with validation criteria
    |
ml-subagent-dev
    Execute each subtask: unit test → implement → Validation Pyramid
    |
training-handoff
    Route between:
    ├── watchdog         (single-run supervision; env restart + async eval)
    └── ml-iteration     (N-round Supervisor-driven iteration against review_criteria)
    |
verification
    Evidence-based conclusion: effective / ineffective / inconclusive

Auto Research (parallel entry for metric search):
    autoresearch-create → ml-brainstorming(autoresearch) → experiment-planning
                       → ml-subagent-dev → autoresearch-handoff → autoresearch
```

- [ ] **Step 4: Add a new "ml-iteration" section right before "Auto Research"**

```markdown
### ml-iteration

`ml-iteration` is the default post-handoff path for "training runs but isn't finished yet." Each round a Researcher subagent modifies code (speed, logging, metric, whatever the user aims it at); the Supervisor runs training + eval, produces a compound review against `review_criteria`, and commits or rolls back. The human stays on the loop — they watch, interject, override, re-aim — but do not gate each round.

**Compound criteria.** Review runs across multiple dimensions collected at brainstorming time:

| Dimension | Examples |
|---|---|
| `metrics` | accuracy ≥ 0.85, loss ≤ 0.3 |
| `performance` | first_step_time ≤ 30s, MFU ≥ 0.30 |
| `observability` | per-step logs include loss/grad_norm/step_time |
| `stability` | no NaN, no torch autograd warnings |
| `custom` | checkpoint format compatible with HF AutoModel |

**Differs from Auto Research.** Auto Research optimizes a single metric against a rigid Fixed/Variable file partition; ml-iteration reviews multi-dimensional criteria with LLM judgment and keeps the human able to override anything. When the experiment is a genuine metric search, pick Auto Research; when it's general iteration-to-ship, pick ml-iteration.
```

- [ ] **Step 5: Update the Skills table**

Locate the ML Workflow skills table. Add:

| Skill | Purpose |
|-------|---------|
| **ml-iteration** | N-round Supervisor-driven iteration against compound review_criteria; Researcher subagent each round; human on the loop |

And update watchdog's row:

| **watchdog** | Single-run training supervision: checkpoint-restart on env failures, async eval, baseline-deviation alerts |

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs(readme): add ml-iteration, narrow watchdog description"
```

---

### Task 15: Update `RELEASE-NOTES.md` and bump version

**Files:**
- Modify: `RELEASE-NOTES.md`
- Modify: `plugin.json`

- [ ] **Step 1: Read current version**

Run: `grep '"version"' plugin.json`
Expected: current minor version (e.g. `"version": "0.30.3"`).

- [ ] **Step 2: Bump version**

Edit `plugin.json` — bump the minor version (feature addition). E.g., `0.30.3 → 0.31.0`.

- [ ] **Step 3: Prepend release notes entry**

Insert at the top of `RELEASE-NOTES.md`:

```markdown
## v0.31.0 (2026-04-13)

### Added

**`ml-iteration` skill — N-round human-on-the-loop iteration**

New post-handoff path for training that runs but isn't finished yet. Each round a Researcher subagent modifies code; Supervisor runs training + eval and produces a compound review against the design-doc's `review_criteria` (metrics, performance, observability, stability, custom). Default commit; rollback on clear regression; human can interject any time.

**`ml-brainstorming` now collects `review_criteria`**

Experiment design docs get a required `review_criteria` block. Fields cover eval metrics, speed expectations, log quality, stability, and anything else discussed. Consumed by `ml-iteration` and (optionally) by `verification`.

**`training-handoff` routes between watchdog and ml-iteration**

After VP, the skill asks the user which path to take. Default suggestion depends on whether VP already satisfies `review_criteria`.

**Shared primitive reference docs at `skills/_ml-loop-primitives/`**

Five small documents (`researcher-dispatch`, `scheduling-safety-net`, `git-control`, `experiences-log`, `eval-lock`) capture patterns shared by `ml-iteration`, `autoresearch`, and (partially) `watchdog`. Not user-invokable; cited by skills.

### Changed

**`watchdog` narrowed to single-run stability**

Removed Tier 2 (parameter auto-fix), Tier 3 (code-fix sub-agent), and the three-mode system (Monitor / Guardian / Autonomous). Kept Tier 1 (env restart from checkpoint), async evaluation, and baseline-deviation alerts. Users who relied on the removed behavior should choose `ml-iteration` at handoff time instead.

### Unchanged

`autoresearch`, `autoresearch-create`, `autoresearch-handoff` are unchanged. Metric-search workflows are not affected.

---

```

- [ ] **Step 4: Commit**

```bash
git add plugin.json RELEASE-NOTES.md
git commit -m "chore: bump version to 0.31.0 and release notes"
```

---

## Self-Review Checklist

Before handing off to execution, walk through these in order:

- [ ] Every file listed in "File Structure" has a task that creates or modifies it.
- [ ] `ml-iteration/SKILL.md` is fully populated (overview, hard gates, startup, main loop, steps 0–9, anomaly, final report) across Tasks 8–11.
- [ ] `training-handoff/SKILL.md` modifications cover both watchdog branch preservation AND iteration branch addition.
- [ ] Watchdog narrowing removes all references to Tier 2 / Tier 3 / three-mode system (Task 6 Step 8 grep confirms this).
- [ ] `review_criteria` schema in `ml-brainstorming` (Task 7) matches the one consumed by `training-handoff` (Task 12) and `ml-iteration` (Task 10).
- [ ] No task references a skill, file, or primitive that isn't created elsewhere in the plan.
- [ ] Version bump in Task 15 matches the semver convention (minor for new features).
- [ ] README workflow diagram and skills table additions match the actual skill file names.
