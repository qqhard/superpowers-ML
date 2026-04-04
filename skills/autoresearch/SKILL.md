---
name: autoresearch
description: Use when running an automated research loop — reads protocol, dispatches Researcher subagent, runs eval + compliance, manages git state, accumulates experience across iterations
---

# Autoresearch

## Overview

Automated research supervisor. Reads a research protocol, then iterates: dispatch Researcher (design + code + train) → Supervisor runs compliance check + evaluation → commit or rollback → repeat until termination.

**Core principle:** Keep the loop running. The Supervisor manages process, evaluation, and git — it does NOT design strategies or write code. Researcher does the creative work; the Supervisor controls everything else.

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

The loop is autonomous. After each round, IMMEDIATELY proceed to the next round. Do NOT wait for user input. Do NOT ask the user questions. Do NOT stop to summarize or suggest options. The only reasons to exit the loop are the termination conditions in Step 4.
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
TaskCreate: "R{round}: Researcher"          activeForm: "Designing strategy + modifying code"
TaskCreate: "R{round}: Compliance check"
TaskCreate: "R{round}: Train"               activeForm: "Training"
TaskCreate: "R{round}: Evaluation"
TaskCreate: "R{round}: Git"
TaskCreate: "R{round}: Termination check"
```
</HARD-GATE>

### Step 1: Dispatch Researcher (design + code only)

Dispatch a fresh subagent with `run_in_background: true`. Researcher designs the strategy and modifies code, but does NOT run training. Supervisor injects all protocol info directly — **Researcher does NOT read the protocol file.**

```
You are an ML researcher. Your task is to improve {metric} ({direction}).
This is Round {round} of {max_rounds}.

## Constraints (from protocol — do NOT read the protocol file)
- **Fixed files (do NOT modify):** {fixed_files}
- **Variable files (you may modify):** {variable_files}
- **Variable range:** {variable_range}

## Your task
1. Read {experiences_path} to learn from past rounds (table format — last {N} rounds shown)
2. Add a row to the experiences table with your strategy (leave Result/Verdict/Insight blank — Supervisor fills those)
3. Modify ONLY the variable files listed above
4. Report "Code ready" as your final message

Do NOT run training. Do NOT run evaluation. Do NOT modify fixed files. Do NOT touch git.
```

### Step 2: Compliance Check

Supervisor runs directly:

```bash
git diff --name-only HEAD
```

Check if ALL changed files are in Variable.files. If any fixed file was modified → round is `not_improved`, skip training and evaluation, go directly to Step 5 (rollback).

### Step 3: Train

Supervisor runs training directly in background. User can `ctrl+o` to see stdout.

```
Bash(
  command: "{train_command}",
  run_in_background: true,
  timeout: {time_limit_ms * 2}   // e.g., time_limit=5min → timeout=600000
)
```

REPL stays idle while training runs — user can interact. Supervisor is notified when training completes (or times out). If timeout, handle as anomaly (record failure, next round Step 0).

### Step 4: Evaluation

Supervisor runs directly:

```bash
{eval_command}  # from protocol's Eval.command
```

Parse the metric value from output. Compare against current best in experiences.md.

### Step 5: Act on Result

**If improved:**
```bash
cp experiences.md /tmp/experiences_backup.md
git add -A
git commit -m "autoresearch: round {round} — {metric}={value} (improved)"
```
Update experiences.md: fill in Result/Verdict/Insight for this round, update best header.

**If not_improved (or compliance failed):**
```bash
cp experiences.md /tmp/experiences_backup.md
git checkout -- .
git clean -fd
cp /tmp/experiences_backup.md experiences.md
```
Update experiences.md: fill in Result/Verdict/Insight for this round.

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

1. Update experiences table: strategy = "agent_error", result = "—", verdict = "❌ error"
2. Rollback: `git checkout -- . && git clean -fd`, restore experiences.md
3. Retry the round ONCE. If retry also fails, skip and continue.
4. **Return to Step 0** — always re-create the task list after anomaly recovery.

### Session Interruption Recovery

On startup, if `experiences.md` shows `status: running` with rounds > 0:
1. Verify git HEAD matches latest committed improvement
2. Last round has no verdict → mid-round interruption, rollback, restart that round
3. Last round has verdict → continue from next round

### Consecutive Failures

5 consecutive not_improved → plateau warning, continue running.

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
