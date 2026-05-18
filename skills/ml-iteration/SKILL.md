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

<HARD-GATE>
### Step 0: Create Round Task List

Create the task list BEFORE dispatching Researcher, for EVERY round.

**Self-check (anti-laziness):** if mid-loop you notice you dispatched Researcher without an S1–S6 list, that is a protocol violation. Stop, build the list now, record the violation in `experiences.md` Insight, continue.

Clear previous round's tasks, then create:

```
TaskCreate: "R{round} S1: Researcher — {task_description from user hint or default}"
TaskCreate: "R{round} S2: Compliance — files + parity ({N} kernels)" if protocol has kernel_targets,
            else "R{round} S2: Compliance — verify locked_files untouched"
TaskCreate: "R{round} S3: Train — {train_command} (limit: {time_limit})"
TaskCreate: "R{round} S4: Eval — {eval_command}"
TaskCreate: "R{round} S5: Review — compound verdict vs review_criteria"
TaskCreate: "R{round} S6: Termination — {round}/{max_rounds}"
```

**Mapping to the detailed step sections below:** S1→Step 1, S2→Step 2, S3→Step 3, S4→Step 4, S5→Step 5, S6→Step 8. Steps 6 (act on verdict), 7 (absorb user input), and 9 (report progress) are handled inline as part of the round and do not require separate task entries.

Update with actual results on completion. S2 completion forms when `kernel_targets` is non-empty:

- S2 → `"R{round} S2: Compliance — ✅"`, `"❌ files: touched {file}"`, or `"❌ parity({target_name}): {kind} {detail snippet}"`

Other rows: strategy summary (S1), verdict (S5), metric snapshot (S4).
</HARD-GATE>

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

## Profile-first discipline (only include this section if `Profile.command` is set in the protocol)
Before designing any strategy, you MUST:
1. Run `{profile_command}` and capture stdout.
2. Save raw profile to `profiles/round-{round}-before.md`.
3. Identify the top hotspot ops/kernels. Save your analysis to
   `profiles/round-{round}-analysis.md` (2-3 sentences: which op
   dominates, by how much, what you suspect).
4. The Strategy you append to experiences.md MUST cite a specific
   hotspot from the analysis. Skipping profile artifacts will be
   flagged by Supervisor and visible to next round's Researcher.

"It's obvious what's slow" is exactly the blind guessing this
discipline prevents.

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
(If perf-mode is off, drop step 1 and renumber.)
1. Run `{profile_command}`, save profile + analysis (see Profile-first discipline above).
2. Read relevant files.
3. Design a strategy for this round (cite a specific hotspot from your profile analysis if in perf mode).
4. Append a row to {experiences_path} with Strategy filled; leave Verdict and Insight blank.
5. Modify code accordingly. You may run quick smoke tests.
6. Report "Code ready" as your final message.
```

Dispatch with `run_in_background: true`. Create two timers (check-in + per-round timeout). Save their IDs.

When the subagent returns, delete both timers and proceed to Step 2.

### Step 2: Compliance Check

Supervisor runs:

```bash
git diff --name-only HEAD
```

Any `locked_files` modified → round is a violation: record in `experiences.md` Insight ("modified locked file {path}"), skip Step 3, Step 4, and Step 5, go to Step 6 with rollback verdict.

Also grep newly-created files for eval-like function names (`evaluate`, `compute_metrics`, `accuracy`, `score`) and flag matches for user review; this is advisory, not a hard block.

If the protocol has a non-empty `kernel_targets` block, run a parity check for each target. For each target's fields from the protocol, execute this Python (substituting the `{{...}}` placeholders with the target's `name`, `new_module`, `baseline_module`, `fixture`, `tolerance.atol`, `tolerance.rtol`, and the experiment directory):

```bash
python3 <<'PY'
import importlib, inspect, sys, torch

NAME = "{{name}}"
NEW = "{{new_module}}"
BASE = "{{baseline_module}}"
FIX = "{{fixture}}"
ATOL, RTOL = {{atol}}, {{rtol}}
SEARCH = "{{experiment_dir}}"

sys.path.insert(0, SEARCH)

def load(spec):
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)

new_fn, base_fn, fixture = load(NEW), load(BASE), load(FIX)

if str(inspect.signature(new_fn)) != str(inspect.signature(base_fn)):
    print(f"PARITY_FAIL target={NAME} kind=signature detail=new{inspect.signature(new_fn)} vs baseline{inspect.signature(base_fn)}", file=sys.stderr); sys.exit(1)

args, kwargs = fixture()
out_new, out_base = new_fn(*args, **kwargs), base_fn(*args, **kwargs)

if not (isinstance(out_new, torch.Tensor) and isinstance(out_base, torch.Tensor)):
    print(f"PARITY_FAIL target={NAME} kind=shape detail=non-tensor output", file=sys.stderr); sys.exit(1)
if out_new.shape != out_base.shape:
    print(f"PARITY_FAIL target={NAME} kind=shape detail=shape {tuple(out_new.shape)} vs {tuple(out_base.shape)}", file=sys.stderr); sys.exit(1)
if out_new.dtype != out_base.dtype:
    print(f"PARITY_FAIL target={NAME} kind=shape detail=dtype {out_new.dtype} vs {out_base.dtype}", file=sys.stderr); sys.exit(1)

try:
    torch.testing.assert_close(out_new, out_base, atol=ATOL, rtol=RTOL)
except AssertionError as e:
    print(f"PARITY_FAIL target={NAME} kind=numerical detail={str(e).splitlines()[0][:200]}", file=sys.stderr); sys.exit(1)

print(f"PARITY_OK target={NAME}")
PY
```

Exit 1 from any target → round is `parity_violation`: record the stderr line in this round's experiences.md Insight, skip Step 3, Step 4, and Step 5; go to Step 6 with rollback verdict. Exit 0 from all targets → proceed.

### Step 2.5: Profile-Discipline Check (advisory)

Only when perf-mode is active (`Profile.command` non-empty in protocol):

```bash
ls profiles/round-{round}-before.md profiles/round-{round}-analysis.md
```

If either file is missing, append to this round's `experiences.md` Insight (do NOT rollback):

> `perf mode but no profile artifacts found — Researcher skipped discipline`

This flag is visible to next round's Researcher when they read the experiences table snippet. It is advisory, not a compliance failure.

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
