---
name: autoresearch
description: Use when running an automated research loop — reads protocol, dispatches Researcher subagent, runs eval + compliance, manages git state, accumulates experience across iterations
---

# Autoresearch

## Overview

Automated research supervisor. Reads a research protocol, then iterates: dispatch Researcher (design + code + train) → Supervisor runs compliance check + evaluation → commit or rollback → repeat until termination.

**Core principle: Human on the Loop.** The loop runs autonomously — the human monitors, not approves. They see every round's result via Task List, inject guidance via Note column, review history via experiences.md and git. The Supervisor keeps the loop running; the human steers from above.

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

Three-layer mechanism:

1. **Researcher notification** — dispatch with `run_in_background: true`. When Researcher completes, the Supervisor is automatically notified and continues.
2. **Per-round timeout timer** — before dispatching Researcher, create a one-shot CronCreate with timeout = `time_limit * 2`. If Researcher doesn't complete before the timer fires, the round is failed. Supervisor records failure, reflects, moves to next round. Delete the timer when Researcher completes normally.
3. **Session heartbeat** — a recurring CronCreate (every 30 minutes) as the ultimate safety net. Only fires when REPL is idle and no task is running.

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
     prompt: "Autoresearch heartbeat — check your loop status."
   )
   ```
   Save the job ID for cleanup.
7. **Enter main loop**

<HARD-GATE>
## Main Loop

The loop is autonomous. Never stop unless the user explicitly says to stop, or termination conditions in Step 6 are met. Do NOT proactively pause, ask questions, or wait for approval.

**User input during the loop:** The user may send messages at any time. Handle based on intent:
- **Stop command** ("停", "stop", "pause") → stop the loop after the current step completes
- **Protocol change** ("model.py 也可以改", "加大 max_rounds") → update autoresearch-protocol.md directly, continue the loop. Supervisor does not proactively modify protocol; Researcher cannot modify protocol. Only user-directed changes.
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

**First, clear previous round's tasks** (if any) — delete all tasks from the previous round so the list stays clean.

**Then create 6 tasks for this round:**
```
TaskCreate: "R{round} S1: Researcher"
TaskCreate: "R{round} S2: Compliance"
TaskCreate: "R{round} S3: Train"
TaskCreate: "R{round} S4: Eval"
TaskCreate: "R{round} S5: Git"
TaskCreate: "R{round} S6: Termination"
```

**Update tasks with results as each step completes** — the task list is the user's primary status view. Use TaskUpdate to enrich subject with outcome:

- Researcher done → subject: `"R{round}: Researcher — {strategy summary}"`
- Compliance done → subject: `"R{round}: Compliance — ✅"` or `"❌ touched {file}"`
- Train done → subject: `"R{round}: Train — {duration}"`
- Eval done → subject: `"R{round}: Eval — {metric}={value} (best: {best} {'↑' or '—'})"`
- Git done → subject: `"R{round}: Git — committed"` or `"rolled back"`
- Termination → subject: `"R{round}: Termination — continue ({round}/{max})"` or `"done"`
</HARD-GATE>

### Step 1: Dispatch Researcher (design + code only)

Dispatch a fresh subagent with `run_in_background: true`. **Supervisor injects lightweight context into the prompt** (constraints + recent experiences). File contents are NOT injected — Researcher reads them itself (one Read call, stays in Researcher's context, not Supervisor's).

Before dispatching, Supervisor extracts from experiences.md: summary + last N rounds table.

```
You are an ML researcher. Your task is to improve {metric} ({direction}).
This is Round {round} of {max_rounds}.

## Your role
You design the strategy and write the code. Training and evaluation are handled by Supervisor after you finish — you don't need to run them. You may run quick smoke tests to verify your code works.

## Constraints
- **Fixed files (do not modify):** {fixed_files}
- **Variable files (you may modify):** {variable_files}
- **Variable range:** {variable_range}
- You may create new files if needed.

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

The training script (framework code, Fixed layer) owns timeout: it saves checkpoint before `time_limit` and exits cleanly. The Bash timeout (+30s buffer) is a fallback — only fires if the script's termination logic fails. REPL stays idle — user can interact. If Bash timeout fires, handle as anomaly.

### Step 4: Evaluation

Supervisor runs directly:

```bash
{eval_command}  # from protocol's Eval.command
```

Parse the metric value from output. Compare against current best in experiences.md.

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
