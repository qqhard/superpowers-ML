---
name: validation-pyramid
description: Use when validating ML training code correctness - orchestrates 3-level checks from static analysis through end-to-end pipeline, integrated into the subagent-dev review workflow
---

# Validation Pyramid

## Overview

The Validation Pyramid ensures ML code is correct before committing to expensive training. Three levels of validation, integrated into the subagent-driven-development workflow after standard code reviews. Each level catches a different class of errors, from cheap/fast (L0) to more thorough (L2).

**Core principle:** In ML, code running without errors does NOT mean it's correct. The Validation Pyramid catches implementation errors so you can trust that "not working" means the strategy is ineffective — not that the code is wrong.

**This is a RIGID skill.** Follow exactly. Don't skip levels. Don't adapt away discipline.

## When to Use

- After implementing any ML code (model, training loop, data pipeline, custom layer)
- During each subtask in an experiment plan
- When diagnostics identifies an issue and you need to re-validate after fixing

## Architecture

The VP runs as 3 stages after standard Superpowers code reviews in the `spml:subagent-dev` workflow:

```
Subagent-Driven-Development (per task)
├─ Implementer writes code
├─ Spec Reviewer (unchanged)
├─ Code Quality Reviewer (unchanged)
├─ L0: VP Static Checks (spml:vp-static-checks)        — static analysis
├─ L1: ML Runtime Validator (spml:ml-runtime-validator) — minutes-level run
├─ L2: ML E2E Validator (spml:ml-e2e-validator)        — pipeline flow check
└─ All pass → task complete
```

Execution order: L0 must pass before L1. L1 must pass before L2.

## Level Summary

| Level | Skill | What it catches | Duration |
|-------|-------|----------------|----------|
| L0 | spml:vp-static-checks | Static config errors, logging & observability (device, precision, optimizer, DataLoader, loss/speed output) | Seconds (code review) |
| L1 | spml:ml-runtime-validator | Performance anomalies (low MFU, gradient NaN, loss not decreasing) | ~5 minutes |
| L2 | spml:ml-e2e-validator | Pipeline flow errors (shape mismatch, checkpoint bug, eval crash) | ~2 minutes |

## Dispatch Model

- **L0** runs as a **subagent** (the vp-static-checks agent, dispatched like spec-reviewer and code-quality-reviewer)
- **L1 and L2** run as **skills invoked by the orchestrator** (execution tasks, not review tasks)

When any level fails, the orchestrator resumes the **same Implementer subagent** to fix the issue. After fixing, re-run only the failed level.

## Shared Fix Loop

All three levels share one mechanism. Each level has its own retry counter (resets when advancing to the next level):

```
Run validation
    → Pass → proceed to next level (reset counter)
    → Fail → send feedback to Implementer with specific issues
        → Implementer fixes → re-run this level
        → 5 consecutive failures at this level → pause, notify user
```

Timeout counts as a failure. Timeouts and metric failures share the same per-level retry counter.

## Large Fix Rollback Rule

If the Implementer's fix modifies more than 50 lines of code, the fix is considered a substantial change. Roll back and re-run all previous stages: Spec Review → Code Quality Review → L0 (and any passed levels before the current one). This prevents large fixes from introducing new problems.

## Red Flags

- Skipping a level because "it's probably fine"
- Running L1 before L0 passes
- Ignoring a failed level and proceeding
- Not re-running after a fix
- "I'll validate later" — validate NOW
- Letting a timeout run instead of killing the process
- Accepting "pass" without checking actual numbers
- Fix > 50 lines without rollback

## Integration

- **spml:brainstorming** — Defines validation scope (which levels, baselines, data flow choice)
- **spml:subagent-dev** — Orchestrates the VP as review stages
- **spml:diagnostics** — Triggered on failure for root cause analysis
- **spml:experiment-planning** — Each subtask specifies which levels apply
