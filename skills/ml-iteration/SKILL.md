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
