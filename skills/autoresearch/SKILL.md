---
name: autoresearch
description: Use when running an automated research loop — reads protocol, dispatches Researcher subagent, runs eval + compliance, manages git state, accumulates experience across iterations
---

# Autoresearch

## Overview

Automated research supervisor. Reads a research protocol, then iterates: dispatch Researcher (design + code + train) → Supervisor runs compliance check + evaluation → commit or rollback → repeat until termination.

**Core principle: Human on the Loop.** The loop runs autonomously — the human monitors, not approves. They see every round's result via Task List, inject guidance via Note column, review history via experiences.md and git. The Supervisor keeps the loop running; the human steers from above.

**Speed principle:** Single-GPU default must guarantee fast first step/epoch print. If baseline is slow, every round is slow. This is solved at baseline construction time — small data, lightweight model, fast first output. Do NOT defer speed optimization to the iteration phase. Baseline speed is a precondition, not an afterthought.

**Programmatic eval principle:** Evaluation MUST be a pre-defined, deterministic script — not code that an agent writes or modifies during the loop. The eval script is fixed before the loop starts (defined during brainstorming, validated during VP L1, extracted by handoff). Humans can run it, agents can run it — same script, same result. This prevents agent self-deception: if the agent could write its own eval, it could (intentionally or not) produce favorable metrics, corrupting the entire research loop. Eval code is always in Fixed.files — Researcher cannot modify it or create alternative eval logic.

**Supervisor's dual role:**
- **Harness maintainer** — Create a reliable execution environment for Researcher. Eval script broken? Fix it. Missing dependency? Install it. .gitignore incomplete? Fill it in. This is infrastructure work that keeps the loop running.
- **Strict executor** — Follow S1→S2→S3→S4→S5→S6 in order. Never skip steps. Never substitute training log metrics for eval. Never skip git operations because "it looks like it failed." Fix issues within the current step, not by skipping it.

These roles do not conflict: maintaining the harness enables strict execution, not bypasses it.

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
Experience transfer happens through files (experiences.md, git history), not agent memory.

**Scheduling (default — Claude Code):**

Four-layer mechanism:

1. **Researcher notification** — dispatch with `run_in_background: true`. When Researcher completes, the Supervisor is automatically notified and continues.
2. **Check-in reminder** — after dispatching ANY background task (Researcher or training), immediately create a one-shot CronCreate to wake yourself up in ~120 seconds (adjust based on expected duration). When it fires: if the task completed, continue; if still running, create another reminder. Delete the reminder when the task completes normally.
3. **Per-round timeout timer** — before dispatching Researcher, create a one-shot CronCreate with timeout = `time_limit * 2`. If Researcher doesn't complete before the timer fires, the round is failed. Supervisor records failure, reflects, moves to next round. Delete the timer when Researcher completes normally.
4. **Session heartbeat** — a recurring CronCreate (every 30 minutes) as the ultimate safety net. Only fires when REPL is idle and no task is running.

**Rule: Never say "I'll wait" without a timer.** If you dispatch a background task and intend to check back later, you MUST create a CronCreate one-shot immediately. Saying "I'll check in 2 minutes" without creating a timer is a bug — there is no built-in mechanism to wake you up at that time. The REPL goes idle and nothing happens until either the task completes (Layer 1) or the 30-minute heartbeat fires (Layer 4). The check-in reminder (Layer 2) fills this gap.

**Scheduling (alternative — non-Claude-Code environments):**

Use a sleep-check loop: dispatch → sleep (estimate from time_limit) → check completion → if not done, halve interval and check again.
</HARD-GATE>

## When to Use

- An autoresearch-protocol.md exists in the experiment directory
- User says "run autoresearch at <dir>"

## Startup

1. **Read `autoresearch-protocol.md`** — extract all fields: research_question, max_rounds, target, baseline, Fixed (files + time_limit + epoch_limit), Variable (files + adjustable range), Eval (metric, direction, command). You will inject these into Researcher's prompt — Researcher does NOT read the protocol file.
2. **Create or reuse worktree:**
   - **Fresh start:** `git worktree add ../autoresearch-{experiment_name} HEAD`
   - **Resume:** Check if worktree exists (`git worktree list`), reuse it.
3. **Verify worktree state:** base code commit exists, `experiences.md` exists.
4. **Check for resume:**
   - Read `experiences.md` header → extract `rounds` and `status`
   - `status: running` + rounds > 0 → resuming, round = rounds + 1
   - `status: not_started` → fresh start, round = 1
   - `status: completed / target_reached` → tell user it's done, ask: continue with more rounds or end?
5. **Announce:** "Starting autoresearch: {research_question}. Round {current}/{max_rounds}. Baseline: {metric} = {baseline}."
6. **Set up session heartbeat:**
   ```
   CronCreate(
     cron: "*/30 * * * *",
     prompt: "Autoresearch heartbeat — self-audit:\n- Current round has S1–S6 task list? If no, you skipped Step 0 — rebuild now.\n- Writing code yourself? Stop — dispatch Researcher subagent.\n- A background task running or round just finished? If neither, you're stalled — resume.\nNever skip steps for \"simplicity\". Then continue."
   )
   ```
   Save the job ID for cleanup.
7. **Enter main loop**

<HARD-GATE>
## Main Loop

The loop is autonomous. Never stop unless the user explicitly says to stop, or termination conditions in Step 6 are met. Do NOT proactively pause, ask questions, or wait for approval.

**User input during the loop:** The user may send messages at any time. Handle based on intent:
- **Stop command** ("stop", "pause") → stop the loop after the current step completes
- **Protocol change** ("allow model.py changes", "increase max_rounds") → update autoresearch-protocol.md directly, continue the loop. Supervisor does not proactively modify protocol; Researcher cannot modify protocol. Only user-directed changes.
- **Guidance / suggestion** → append to current round's Note column, continue the loop
- **Question** → answer briefly, continue the loop

Never stop the loop for anything other than an explicit stop command or termination conditions.
</HARD-GATE>

```
for round in current_round..max_rounds:

  0. CREATE ROUND TASK LIST
  1. DISPATCH RESEARCHER (design + code only)
  2. COMPLIANCE CHECK
  3. TRAIN (Supervisor executes)
  4. EVALUATION
  5. ACT ON RESULT
  6. CHECK TERMINATION
  7. REPORT PROGRESS
```

<HARD-GATE>
### Step 0: Create Round Task List

You MUST create the task list BEFORE dispatching Researcher. This applies to EVERY round, including rounds following anomaly recovery.

**Self-check (anti-laziness):** If mid-loop you notice you dispatched Researcher without building the S1–S6 task list, that is a protocol violation — not a shortcut. Stop the current round, build the task list now, record the violation in this round's experiences.md Insight, then continue. "Context got long so I simplified" is not a valid reason to skip steps.

**First, clear previous round's tasks** (if any) — delete all tasks from the previous round so the list stays clean.

**Then create 6 tasks for this round:**
Create with protocol context, update with actual results:

```
TaskCreate: "R{round} S1: Researcher — improve {metric} on {variable_files}"
TaskCreate: "R{round} S2: Compliance — check {variable_files} only"
TaskCreate: "R{round} S3: Train — {train_command} (limit: {time_limit})"
TaskCreate: "R{round} S4: Eval — {eval_command} (current best: {best_value})"
TaskCreate: "R{round} S5: Git — awaiting result"
TaskCreate: "R{round} S6: Termination — {round}/{max_rounds}"
```

**Update with actual results on completion:**

- S1 → `"R{round} S1: Researcher — {strategy summary, e.g. 'cosine lr + label smoothing'}"`
- S2 → `"R{round} S2: Compliance — ✅"` or `"❌ touched {file}"`
- S3 → `"R{round} S3: Train — done {duration}, final loss={value}"`
- S4 → `"R{round} S4: Eval — {metric}={value} (best: {best} {'↑' or '—'})"`
- S5 → `"R{round} S5: Git — committed"` or `"rolled back"`
- S6 → `"R{round} S6: Termination — continue"` or `"done: {reason}"`
</HARD-GATE>

### Step 1: Dispatch Researcher (design + code only)

Dispatch a fresh subagent with `run_in_background: true`. **Supervisor injects lightweight context into the prompt** (constraints + recent experiences). File contents are NOT injected — Researcher reads them itself (one Read call, stays in Researcher's context, not Supervisor's).

Before dispatching, Supervisor extracts from experiences.md: summary + last N rounds table.

**After dispatching, immediately create TWO timers:**
1. Check-in reminder (~120s): `CronCreate(schedule: "120s", prompt: "Check-in: Researcher round {round} — verify completion or check progress.")`
2. Per-round timeout (`time_limit * 2`): as described in Layer 3.

Save both IDs. When Researcher completes normally, delete both timers before continuing to Step 2.

```
You are an ML researcher. Your task is to improve {metric} ({direction}).
This is Round {round} of {max_rounds}.

## Your role
You design the strategy and write the code. Training and evaluation are handled by Supervisor after you finish — you don't need to run them. You may run quick smoke tests to verify your code works.

## Constraints
- **Fixed files (do not modify):** {fixed_files}
- **Variable files (you may modify):** {variable_files}
- **Variable range:** {variable_range}
- You may create new helper files if needed.
- **Do NOT create or modify any evaluation logic.** Evaluation is a pre-defined script managed by Supervisor. Do not write alternative eval scripts, metric computation code, or accuracy calculation utilities — even in new files.

## Recent experiences (last {N} rounds)
{experiences_table_snippet}

## Your task
1. Read the variable files listed above to understand current code
2. Based on experiences above, design a strategy for this round
3. Add a row to {experiences_path} with your strategy (leave Result/Verdict/Insight blank)
4. Modify the variable files to implement your strategy
5. Report "Code ready" as your final message
```

### Step 2: Compliance Check

Supervisor runs directly:

```bash
git diff --name-only HEAD
```

Check that no Fixed.files were modified. New files are allowed. If any fixed file was modified → round is `not_improved`, skip training and evaluation, go directly to Step 5 (rollback).

### Step 3: Train

Supervisor runs training directly in background. User can `ctrl+o` to see stdout.

```
Bash(
  command: "{train_command}",
  run_in_background: true,
  timeout: {time_limit_ms + 30000}   // time_limit + 30s buffer, e.g., 5min → 330000
)
```

**Immediately after starting training, create a check-in reminder:**
`CronCreate(schedule: "{time_limit * 0.8}s", prompt: "Check-in: Training round {round} — verify completion or check progress.")`
Use ~80% of time_limit as the first check-in interval (e.g., 5min limit → 240s reminder). Save the ID and delete it when training completes.

The training script (framework code, Fixed layer) owns timeout: it saves checkpoint before `time_limit` and exits cleanly. The Bash timeout (+30s buffer) is a fallback — only fires if the script's termination logic fails. REPL stays idle — user can interact. If Bash timeout fires, handle as anomaly.

<HARD-GATE>
### Step 4: Evaluation

Supervisor runs eval_command directly. This is the ONLY source of truth for the metric — training log output does NOT count. Do NOT skip this step or substitute training log metrics.

**Programmatic eval enforcement:** The eval_command is a fixed, pre-defined script from the protocol. Supervisor runs it as-is — no modification, no wrapping, no "improved" version. If Researcher created any new eval scripts or modified eval logic (even in new files), those are compliance violations — ignore them and use the original eval_command only.

```bash
{eval_command}  # from protocol's Eval.command — NEVER substituted
```

Parse the metric value from output. Compare against current best in experiences.md. If eval_command fails, fix the environment (missing deps, wrong paths) but NEVER change the eval logic itself — do NOT fall back to training log metrics.
</HARD-GATE>

### Step 5: Act on Result

**First, ensure .gitignore covers training artifacts** (outputs, logs, checkpoints, etc.). Verify this once during Startup — if .gitignore is missing or incomplete, fix it before the loop starts. With a proper .gitignore, `git add -A` naturally skips artifacts.

**If improved:**
```bash
cp experiences.md /tmp/experiences_backup.md
git add -A
git commit -m "autoresearch: round {round} — {metric}={value} (improved)"
```
Update experiences.md: fill in Result/Verdict, update best header. Insight: what worked and why.

**If not_improved (or compliance failed):**
```bash
cp experiences.md /tmp/experiences_backup.md
git checkout -- .
git clean -fd
cp /tmp/experiences_backup.md experiences.md
```
Update experiences.md: fill in Result/Verdict. Insight MUST explain why it failed — this guides the next round's Researcher.

<HARD-GATE>
### Step 6: Check Termination

The Supervisor terminates ONLY for these exact reasons. No exceptions, no early stops based on judgment.

- If `target` is set AND metric reaches target → `target_reached`
- If `round == max_rounds` → `completed`
- Otherwise: **continue to next round**

The Supervisor does NOT decide "the metric can't improve further." The protocol said how many rounds to run — run them.
</HARD-GATE>

### Step 7: Report Progress

```
Round {round}/{max_rounds}: {metric}={value} — {verdict}
  Best so far: {metric}={best_value} (Round {best_round})
```

## Anomaly Recovery

### Researcher Timeout / Crash

1. Update experiences table with diagnostic Insight:
   - **Timeout**: strategy = Researcher's last known action, insight = "timeout after Xs — likely cause: [Supervisor's diagnosis, e.g., training hung on data loading, infinite loop in augmentation]"
   - **Crash**: strategy = Researcher's strategy if available, insight = "crash: [error message]. Likely cause: [diagnosis]"
   - **Compliance fail**: insight = "modified fixed file {filename} — [what the change was and why it violated protocol]"
2. Rollback: `git checkout -- . && git clean -fd`, restore experiences.md
3. Decide whether to retry based on the failure mode. Transient errors (OOM, network) may warrant a retry; logic errors (wrong API, broken code) won't be fixed by retrying. Use judgment.
4. **Return to Step 0** — always re-create the task list after anomaly recovery.

These diagnostic insights guide the next round's Researcher — it reads the experiences table and should avoid repeating the same mistake.

### Session Interruption Recovery

On startup, if `experiences.md` shows `status: running` with rounds > 0:
1. Verify git HEAD matches latest committed improvement
2. Last round has no verdict → mid-round interruption, rollback, restart that round
3. Last round has verdict → continue from next round

### Consecutive Failures

If multiple consecutive rounds show no improvement, output a plateau warning with analysis of why (e.g., "variable space may be exhausted", "strategies are repeating"). Continue running — do not stop. The warning is informational, not a termination condition.

## Final Report

When the loop terminates:
1. **Delete the heartbeat cron** — `CronDelete` with the job ID
2. Update experiences.md status
3. **Worktree options:** merge / keep / remove

```
# Autoresearch Complete

## Result
- Best: {metric} = {value} (Round {N})
- Baseline: {metric} = {baseline_value}
- Improvement: {delta}
- Rounds: {completed} / {max_rounds}
- Termination: {reason}

## Key Insights
<Distill top insights from experiences table>
```
