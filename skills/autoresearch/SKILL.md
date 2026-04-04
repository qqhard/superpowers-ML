---
name: autoresearch
description: Use when running an automated research loop — reads protocol, dispatches designer/evaluator subagents, manages git state, accumulates experience across iterations
---

# Autoresearch

## Overview

Automated research supervisor. Reads a research protocol, then iterates: dispatch Agent A (design + code + train) → dispatch Agent B (evaluate + review + record) → commit or rollback based on verdict → repeat until termination.

**Core principle:** Keep the loop running. The Supervisor manages form and process — it does NOT design strategies or write code. Agent A and Agent B do the creative work; the Supervisor controls git, checks termination, and recovers from failures.

<HARD-GATE>
## Git Control

You MUST be the ONLY entity that performs git write operations (commit, checkout, reset).
Agent A and Agent B have file read/write and bash permissions but NO git write permissions.
All git state changes go through the Supervisor.
All git operations happen in the worktree — NEVER in the main working directory.
</HARD-GATE>

<HARD-GATE>
## Monitoring Loop Mechanism

You MUST use subagents (Agent tool) as the ONLY way to dispatch Agent A and Agent B.
Each agent is a fresh subagent per round — no shared context between rounds.
Experience transfer happens through files (experiences.md, git history), not agent memory.

**Non-blocking dispatch:** Always use `run_in_background: true` for Agent A and Agent B. The Supervisor stays idle while agents work — this prevents session timeout during long training runs and keeps the REPL responsive. You will be automatically notified when each agent completes. Do NOT poll or sleep-wait for agents.
</HARD-GATE>

## When to Use

- User has pasted an autoresearch prompt from autoresearch-handoff
- An autoresearch-protocol.md exists in the experiment directory

## Startup

1. **Read `autoresearch-protocol.md`** — understand research question, fixed/variable/pressure conditions, evaluation method, termination criteria, agent boundary
2. **Create or reuse worktree** — all autoresearch operations happen in an isolated worktree, never in the main working directory.
   - **Fresh start:** Create a new worktree from the current HEAD:
     ```bash
     git worktree add ../autoresearch-{experiment_name} HEAD
     ```
     Record the worktree path. All subsequent `{worktree_path}` references point here.
   - **Resume:** Check if the worktree already exists (from a previous session). If so, reuse it — do NOT create a new one. Verify with `git worktree list`.
   - Copy `autoresearch-protocol.md` and `experiences.md` into the worktree if they aren't already there (they live in the experiment directory within the worktree).
3. **Verify worktree state** (inside the worktree):
   - Base code commit exists (`git log --oneline -1`)
   - `experiences.md` exists and has Summary section
4. **Check for resume:**
   - Read `experiences.md` Summary → extract `Total rounds` and `Status`
   - If Status is `running` and Total rounds > 0: resuming from interruption
     - Current round = Total rounds + 1
     - Verify git HEAD matches latest committed improvement (or baseline if no improvements yet)
   - If Status is `not_started`: fresh start, round = 1
5. **Announce:** "Starting autoresearch: {research_question}. Round {current} / {max_rounds}. Baseline: {metric} = {baseline}. Worktree: {worktree_path}."
6. **Set up heartbeat reminder** — create a session-scoped recurring CronCreate that fires every 30 minutes:
   ```
   CronCreate(
     cron: "*/30 * * * *",
     prompt: "Autoresearch heartbeat: you are running an autoresearch loop at {experiment_dir}. Check experiences.md — is the loop still progressing? If you have a background agent running, check on it. If the loop stalled, resume from where you left off."
   )
   ```
   Save the returned job ID — you need it for cleanup.
   
   **Why:** The Supervisor is a language model, not a persistent process. Periodic reminders keep it aware that a loop is active. When the reminder fires, the Supervisor checks its own state and re-activates the loop if needed. This is a simple ping, not complex state detection.
7. **Enter main loop**

<HARD-GATE>
## Main Loop

The loop is autonomous. After each round, IMMEDIATELY proceed to the next round. Do NOT wait for user input. Do NOT ask the user questions. Do NOT stop to summarize or suggest options. The only reasons to exit the loop are the termination conditions in Step 4.
</HARD-GATE>

```
for round in current_round..max_rounds:

  0. CREATE ROUND TASK LIST
  1. DISPATCH AGENT A
  2. DISPATCH AGENT B
  3. ACT ON VERDICT
  4. CHECK TERMINATION
  5. REPORT PROGRESS
```

### Step 0: Create Round Task List

At the start of each round, create a task list so the user can track progress:

```
TaskCreate: "Round {round}/{max_rounds}"
  subtasks:
    - "Agent A: design + code + train"
    - "Agent B: evaluate + review + record"
    - "Git: commit or rollback based on verdict"
    - "Check termination"
```

Mark each subtask as completed when the corresponding step finishes.

### Step 1: Dispatch Agent A

Dispatch a fresh subagent with the following prompt structure. Agent A has full file read/write and bash access but NO git write access.

```
You are an ML researcher. Your task is to improve {metric} ({direction}).
This is Round {round} of {max_rounds}.

## File Paths
- Protocol: {protocol_path}  — read this for research question, constraints, and your degrees of freedom
- Experience log: {experiences_path}  — read this for all past strategies and results
- Codebase: {worktree_path}

## Instructions
- Read the protocol FIRST. Pay close attention to:
  - **Fixed Conditions**: things you MUST NOT change. Violating these invalidates the round.
  - **Variable Conditions**: things you CAN change. Your creativity is limited to this space.
- Read experiences.md to learn from past rounds — decide yourself how much to read
- Git history contains successful strategies as commits — use git log / git diff to study what code changes worked
- Do NOT modify termination logic, evaluation logic, or anything listed as Fixed Conditions
- Write your strategy description to strategy.md BEFORE modifying any code. Explicitly state which Variable Conditions you are changing and why.
- After modifying code, run training: the pressure conditions will auto-terminate at the limit
- When training completes, report "Training complete" as your final message
```

Dispatch with `run_in_background: true`. The Supervisor does NOT block — REPL stays idle until notified that Agent A has completed. If Agent A times out or crashes, see Anomaly Recovery.

### Step 2: Dispatch Agent B

After receiving Agent A's completion notification, dispatch a fresh subagent. Agent B has full file read/write and bash access but NO git write access.

```
You are an ML experiment evaluator and reviewer. Your task is to independently evaluate Round {round}'s result and audit Agent A's strategy for protocol compliance.

## File Paths
- Protocol: {protocol_path}  — read this for evaluation method, metric details, Fixed Conditions, and Variable Conditions
- Experience log: {experiences_path}  — read this for current best result in Summary section
- Strategy: {worktree_path}/strategy.md  — read this for what Agent A claims to have done
- Codebase: {worktree_path}

## Instructions

### Part 1: Protocol Compliance Audit
1. Read the protocol's Fixed Conditions and Variable Conditions
2. Read Agent A's strategy.md
3. Review the actual code changes (use git diff HEAD~1 or compare against known baseline) to verify:
   - Agent A did NOT violate any Fixed Conditions
   - Agent A only modified things listed in Variable Conditions
   - The strategy described in strategy.md matches the actual code changes
4. If Agent A violated Fixed Conditions, your verdict MUST be not_improved regardless of metrics, and your Insight must explain the violation

### Part 2: Independent Evaluation
5. Run the evaluation command from the protocol's Evaluation section — do NOT read training logs as a substitute
6. Record the metric value from YOUR evaluation run, not from Agent A's training output

### Part 3: Record Experience
7. Read the current best result from experiences.md Summary section
8. Compare this round's result against the best result
9. Append a new Round entry to experiences.md:
   - **Strategy**: (from strategy.md)
   - **Compliance**: ✅ protocol respected / ❌ Fixed Condition violated: {detail}
   - **Result**: {metric} = {value}
   - **Verdict**: ✅ committed or ❌ rolled back
   - **Insight**: what worked or didn't and why (include compliance issues if any)
10. Your final message MUST be exactly one of these two formats:
    VERDICT: improved {metric}={value}
    VERDICT: not_improved {metric}={value}
```

Dispatch with `run_in_background: true`. After receiving Agent B's completion notification, parse the VERDICT line from its response.

### Step 3: Act on Verdict

**If improved:**
```bash
# Save experiences.md first (Agent B already updated it)
cp experiences.md /tmp/experiences_backup.md

# Commit all changes including updated experiences.md
git add -A
git commit -m "autoresearch: round {round} — {metric}={value} (improved)"

# Update Summary section
# (edit experiences.md: update Best result line, increment Total rounds)
```

**If not_improved:**
```bash
# Save experiences.md (Agent B already appended the failed round entry)
cp experiences.md /tmp/experiences_backup.md

# Rollback all code changes
git checkout -- .
git clean -fd

# Restore experiences.md with the new round entry
cp /tmp/experiences_backup.md experiences.md

# Update Summary section
# (edit experiences.md: increment Total rounds only, keep Status as running)
```

<HARD-GATE>
### Step 4: Check Termination

The Supervisor terminates ONLY for these exact reasons. No exceptions, no early stops based on judgment.

- If `target` is set AND (`direction == minimize` and `metric_value <= target`, or `direction == maximize` and `metric_value >= target`): terminate with reason `target_reached`
- If `round == max_rounds`: terminate with reason `completed`
- Otherwise: **continue to next round**

If target is "none" or unset, the ONLY termination condition is reaching max_rounds. The Supervisor does NOT decide "the metric can't improve further" or "we've hit a theoretical ceiling." That is the researcher's judgment to make after reviewing all rounds. The protocol said how many rounds to run — run them.
</HARD-GATE>

### Step 5: Report Progress

After each round, output a status line:

```
Round {round}/{max_rounds}: {metric}={value} — {verdict}
  Best so far: {metric}={best_value} (Round {best_round})
  Trend: {improving / plateauing / oscillating}
```

Trend detection:
- **improving**: at least one of the last 3 rounds was "improved"
- **plateauing**: last 3+ rounds all "not_improved"
- **oscillating**: alternating improved/not_improved in last 4 rounds

## Anomaly Recovery

### Agent Timeout / Crash

If Agent A or Agent B fails (subagent returns error or times out):

1. Record in experiences.md:
   ```
   ## Round {N}
   - **Strategy**: (agent_error — Agent {A/B} failed)
   - **Result**: N/A
   - **Verdict**: ⚠️ skipped (agent error)
   - **Insight**: {error message if available}
   ```
2. Rollback any partial code changes: `git checkout -- . && git clean -fd`
3. Restore experiences.md (same backup/restore pattern as not_improved)
4. Retry the round ONCE. If retry also fails, skip this round and continue to next.

### Session Interruption Recovery

On startup, if `experiences.md` Summary shows `Status: running` with `Total rounds > 0`:
1. Read the last Round entry to understand where we left off
2. Verify git HEAD is consistent (matches latest committed improvement)
3. If the last round entry has no Verdict: the interruption happened mid-round. Treat as failed round, rollback any uncommitted changes, continue from that round number.
4. If the last round has a Verdict: continue from next round.

### Consecutive Failures

If 5 consecutive rounds are "not_improved":
- Output a plateau warning: "⚠️ Plateau detected: 5 consecutive rounds without improvement. Continuing, but consider reviewing the protocol's Variable Conditions for wider exploration."
- Continue the loop (do not stop).

## Final Report

When the loop terminates (target reached or max rounds):
1. **Delete the watchdog cron** — use `CronDelete` with the job ID saved during startup
2. Update experiences.md Summary — set Status to `target_reached` or `completed`
3. **Merge worktree results** — the best code is in the worktree's HEAD. Present the user with options:
   - Merge the worktree branch into main
   - Keep the worktree for manual review
   - Remove the worktree (user will handle results manually)

Then output the final report:

```
# Autoresearch Complete

## Result
- Best: {metric} = {value} (Round {N})
- Baseline: {metric} = {baseline_value}
- Improvement: {delta} ({percentage}%)
- Rounds: {completed} / {max_rounds}
- Termination reason: {target_reached / completed}

## Key Insights
<Read experiences.md and distill top 3-5 insights across all rounds>

## Current State
- git HEAD = best performing code
- All experience recorded in experiences.md
- Worktree: {path}
```

## Integration

- **spml:autoresearch-handoff** — Generates the protocol and startup prompt that triggers this skill
- **spml:ml-brainstorming** — Design doc's Autoresearch Protocol section defines the protocol content
- **spml:ml-subagent-dev** — Post-Completion Gate routes to autoresearch-handoff
