---
name: validation-pyramid
description: Use when validating the assembled training pipeline of an ML experiment - orchestrates 2-level checks (L0 static + L1 runtime) on the single [INTEGRATION] subtask only, exactly once
---

# Validation Pyramid

## Overview

The Validation Pyramid ensures the assembled ML training pipeline is correct before committing to expensive long-running training. Two levels of validation, fired **once per experiment**, on the single `[INTEGRATION]` subtask: L0 catches static config errors cheaply (seconds), L1 catches runtime and pipeline errors thoroughly (minutes).

**Core principle:** In ML, code running without errors does NOT mean it's correct. The Validation Pyramid catches integration errors so you can trust that "not working" means the strategy is ineffective — not that the assembled training pipeline is wrong.

**VP fires once, on integration only.** Pure code subtasks (model class, dataset, loss, evaluator core, etc.) are validated by standard superpowers TDD + spec review + quality review — they do NOT run L0 or L1. The Validation Pyramid only fires when the components are assembled into the final delivered training pipeline.

**VP is validation, not the experiment.** Both levels (L0, L1) run with limited steps/data to verify correctness cheaply. The full training run happens AFTER VP passes, via `spml:training-handoff` → `spml:watchdog`. Never run VP levels with full experiment settings (full epochs, full dataset iterations).

**This is a RIGID skill.** Follow exactly. Don't skip levels. Don't adapt away discipline. Don't fire VP on code subtasks.

## When to Use

- Exactly once per experiment, on the single `[INTEGRATION]` subtask after standard reviews pass
- Invoked from `spml:ml-subagent-dev`'s integration path
- Re-invoked when the integration subtask is revised (`[ ] REVISED`)
- Re-invoked per round inside iterative orchestrators (`spml:ml-iteration`, `spml:autoresearch`) — each round is itself a delivery and re-validates the assembled pipeline

## When NOT to Use

- On any code subtask (model class, dataset, loss, custom layer, evaluator core, etc.). Standard superpowers reviews are sufficient — VP would waste 5-15 minutes per subtask without testing the integration that actually matters.
- During ad-hoc debugging of a single component. Use unit tests + diagnostics instead.

## Architecture

The VP runs as 2 stages **after** standard Spec Review and Code Quality Review pass on the integration subtask, inside `spml:ml-subagent-dev`:

```
ml-subagent-dev (integration subtask only)
├─ Implementer assembles training pipeline + integration tests
├─ Spec Reviewer
├─ Code Quality Reviewer
├─ L0: ML Static Checks (spml:ml-static-checks)            — static analysis
├─ L1: ML Runtime Validation (spml:ml-runtime-validator)   — training + pipeline
└─ All pass → integration subtask complete → Post-Completion Gate
```

Code subtasks complete after Quality Review. They never enter this pipeline.

Execution order: L0 must pass before L1.

## Level Summary

| Level | Skill | What it catches | Duration |
|-------|-------|----------------|----------|
| L0 | spml:ml-static-checks | Static config errors, logging & observability (device, precision, optimizer, DataLoader, loss/speed output) | Seconds (code review) |
| L1 | spml:ml-runtime-validator | Performance anomalies + pipeline flow errors (MFU, gradients, loss, checkpoint, inference, evaluation) | ~5-15 minutes |

## Dispatch Model

- **L0** runs as a **subagent** (the ml-static-checks agent, dispatched after the code-quality reviewer)
- **L1** runs as a **skill invoked by the orchestrator** (execution task, not review task)

When any level fails, the orchestrator resumes the **same Implementer subagent** (the one that built the integration) to fix the issue. After fixing, re-run only the failed level.

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

If the Implementer's fix to the integration subtask modifies more than 50 lines, the fix is considered a substantial change. Roll back and re-run all previous integration stages: Spec Review → Code Quality Review → L0 (and any passed levels before the current one). This prevents large fixes from introducing new problems.

If the fix touches files owned by an upstream code subtask, that code subtask must be re-flagged for re-execution; its unit tests + reviews re-run, and the integration must re-run from the beginning afterwards.

## Red Flags

- Firing VP on a code subtask "to be safe"
- Skipping a level on the integration subtask because "it's probably fine"
- Running L1 before L0 passes
- Ignoring a failed level and proceeding
- Not re-running after a fix
- "I'll validate later" — validate NOW (on the integration subtask)
- Letting a timeout run instead of killing the process
- Accepting "pass" without checking actual numbers
- Fix > 50 lines without rollback
- **Running VP with full experiment settings** (e.g., 80 epochs) — VP is validation, not the experiment
- **Monkey-patching to limit training** instead of using config/CLI flags — monkey-patches fail silently
- **L1 running longer than expected** without investigating — likely means duration limiting failed
- Allowing zero or more-than-one `[INTEGRATION]` subtasks in the plan

## Integration

- **spml:ml-brainstorming** — Defines validation scope (which levels, baselines, data flow choice) for the integration subtask
- **spml:experiment-planning** — Marks exactly one `[INTEGRATION]` subtask; that subtask alone runs VP
- **spml:ml-subagent-dev** — Orchestrates the VP as the final stages of the integration subtask
- **spml:diagnostics** — Triggered on failure for root cause analysis
- **spml:ml-iteration / spml:autoresearch** — Run their own per-round VP (each round IS an integration delivery)
