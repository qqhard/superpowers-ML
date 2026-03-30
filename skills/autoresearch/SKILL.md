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
</HARD-GATE>

<HARD-GATE>
## Monitoring Loop Mechanism

You MUST use subagents (Agent tool) as the ONLY way to dispatch Agent A and Agent B.
Each agent is a fresh subagent per round — no shared context between rounds.
Experience transfer happens through files (experiences.md, git history), not agent memory.
</HARD-GATE>

## When to Use

- User has pasted an autoresearch prompt from autoresearch-handoff
- An autoresearch-protocol.md exists in the experiment directory

## Startup

1. **Read `autoresearch-protocol.md`** — understand research question, fixed/variable/pressure conditions, evaluation method, termination criteria, agent boundary
2. **Verify worktree state:**
   - Base code commit exists (`git log --oneline -1`)
   - `experiences.md` exists and has Summary section
3. **Check for resume:**
   - Read `experiences.md` Summary → extract `Total rounds` and `Status`
   - If Status is `running` and Total rounds > 0: resuming from interruption
     - Current round = Total rounds + 1
     - Verify git HEAD matches latest committed improvement (or baseline if no improvements yet)
   - If Status is `not_started`: fresh start, round = 1
4. **Announce:** "Starting autoresearch: {research_question}. Round {current} / {max_rounds}. Baseline: {metric} = {baseline}."
5. **Enter main loop**

## Main Loop

```
for round in current_round..max_rounds:

  1. DISPATCH AGENT A
  2. DISPATCH AGENT B
  3. ACT ON VERDICT
  4. CHECK TERMINATION
  5. REPORT PROGRESS
```

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
- Read the protocol first to understand what you can and cannot change
- Read experiences.md to learn from past rounds — decide yourself how much to read
- Git history contains successful strategies as commits — use git log / git diff to study what code changes worked
- Do NOT modify termination logic or evaluation logic (these are Fixed Conditions in the protocol)
- Write your strategy description to strategy.md BEFORE modifying any code
- After modifying code, run training: the pressure conditions will auto-terminate at the limit
- When training completes, report "Training complete" as your final message
```

Wait for Agent A to complete. If Agent A times out or crashes, see Anomaly Recovery.

### Step 2: Dispatch Agent B

Dispatch a fresh subagent with the following prompt structure. Agent B has full file read/write and bash access but NO git write access.

```
You are an ML experiment evaluator. Your task is to evaluate Round {round}'s result and record experience.

## File Paths
- Protocol: {protocol_path}  — read this for evaluation method and metric details
- Experience log: {experiences_path}  — read this for current best result in Summary section
- Strategy: {worktree_path}/strategy.md  — read this for what Agent A tried this round

## Instructions
1. Read the protocol's Evaluation section for the eval_command and metric details
2. Run the evaluation command from the protocol
3. Read the current best result from experiences.md Summary section
4. Compare this round's result against the best result
5. Append a new Round entry to experiences.md:
   - **Strategy**: (from strategy.md)
   - **Result**: {metric} = {value}
   - **Verdict**: ✅ committed or ❌ rolled back
   - **Insight**: what worked or didn't and why
6. Your final message MUST be exactly one of these two formats:
   VERDICT: improved {metric}={value}
   VERDICT: not_improved {metric}={value}
```

Wait for Agent B to complete. Parse the VERDICT line from Agent B's response.

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

### Step 4: Check Termination

- Parse the metric value from Agent B's VERDICT line
- If `direction == minimize` and `metric_value <= target`: terminate with reason `target_reached`
- If `direction == maximize` and `metric_value >= target`: terminate with reason `target_reached`
- If `round == max_rounds`: terminate with reason `completed`
- Otherwise: continue to next round

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

When the loop terminates (target reached or max rounds), update experiences.md Summary:
- Set Status to `target_reached` or `completed`

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
