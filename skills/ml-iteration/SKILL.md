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
