---
name: validation-pyramid
description: Use when validating ML training code correctness - orchestrates 2-level checks (L0 static + L1 runtime), integrated into the ml-subagent-dev review workflow
---

# Validation Pyramid

## Overview

The Validation Pyramid ensures ML code is correct before committing to expensive training. Two levels of validation, integrated into the ml-subagent-dev workflow after standard code reviews. L0 catches static config errors cheaply (seconds), L1 catches runtime and pipeline errors thoroughly (minutes).

**Core principle:** In ML, code running without errors does NOT mean it's correct. The Validation Pyramid catches implementation errors so you can trust that "not working" means the strategy is ineffective — not that the code is wrong.

**VP is validation, not the experiment.** Both levels (L0, L1) run with limited steps/data to verify correctness cheaply. The full training run happens AFTER VP passes, via `spml:training-handoff` → `spml:watchdog`. Never run VP levels with full experiment settings (full epochs, full dataset iterations).

**This is a RIGID skill.** Follow exactly. Don't skip levels. Don't adapt away discipline.

## When to Use

- After implementing any ML code (model, training loop, data pipeline, custom layer)
- During each subtask in an experiment plan
- When diagnostics identifies an issue and you need to re-validate after fixing

## Architecture

The VP runs as 2 stages after standard Superpowers code reviews in the `spml:ml-subagent-dev` workflow:

```
Subagent-Driven-Development (per task)
├─ Implementer writes code
├─ Spec Reviewer (unchanged)
├─ Code Quality Reviewer (unchanged)
├─ L0: ML Static Checks (spml:ml-static-checks)            — static analysis
├─ L1: ML Runtime Validation (spml:ml-runtime-validator)   — training + pipeline
└─ All pass → task complete
```

Execution order: L0 must pass before L1.

## Level Summary

| Level | Skill | What it catches | Duration |
|-------|-------|----------------|----------|
| L0 | spml:ml-static-checks | Static config errors, logging & observability (device, precision, optimizer, DataLoader, loss/speed output) | Seconds (code review) |
| L1 | spml:ml-runtime-validator | Performance anomalies + pipeline flow errors (MFU, gradients, loss, checkpoint, inference, evaluation) | ~5-15 minutes |

## Dispatch Model

- **L0** runs as a **subagent** (the ml-static-checks agent, dispatched like spec-reviewer and code-quality-reviewer)
- **L1** runs as a **skill invoked by the orchestrator** (execution task, not review task)

When any level fails, the orchestrator resumes the **same Implementer subagent** to fix the issue. After fixing, re-run only the failed level.

## Shared Fix Loop

Both levels share one mechanism. Each level has its own retry counter (resets when advancing to the next level):

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
- **Running VP with full experiment settings** (e.g., 80 epochs) — VP is validation, not the experiment
- **Monkey-patching to limit training** instead of using config/CLI flags — monkey-patches fail silently
- **L1 running longer than expected** without investigating — likely means duration limiting failed

## Integration

- **spml:ml-brainstorming** — Defines validation scope (which levels, baselines, data flow choice)
- **spml:ml-subagent-dev** — Orchestrates the VP as review stages
- **spml:diagnostics** — Triggered on failure for root cause analysis
- **spml:experiment-planning** — Each subtask specifies which levels apply
