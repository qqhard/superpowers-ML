---
name: ml-static-checks
description: Use when reviewing ML code for static correctness — dispatched after Spec Review and Code Quality Review in the ml-subagent-dev workflow
---

# L0: ML Static Checks

## Overview

A specialized static-analysis subagent that checks ML code correctness and training observability. Runs after standard Spec Review and Code Quality Review pass. Catches configuration errors, device mismatches, precision problems, and optimization oversights — issues that ML agents commonly introduce and that don't require runtime to detect.

**This is a RIGID skill.** Follow the checklist exactly. Don't skip applicable checks.

## When to Use

- Automatically dispatched by the orchestrator in `spml:ml-subagent-dev` **only on the single `[INTEGRATION]` subtask**, after Spec Review and Code Quality Review pass
- Code subtasks (model class, dataset, loss, custom layer, evaluator core, etc.) do NOT run L0 — standard superpowers reviews cover them
- Re-invoked per round inside iterative orchestrators (`spml:ml-iteration`, `spml:autoresearch`) since each round is itself an integration delivery
- Skip for pure infrastructure tasks (CI, docs, config files)

## How It Works

1. Orchestrator dispatches `ml-static-checks` agent (defined in `agents/ml-static-checks.md`)
2. Agent reads all changed files
3. Agent evaluates each checklist item's applicability condition
4. For applicable items: verify the code meets the requirement
5. Report findings using Critical/Important/Suggestion severity levels

## Severity Tiers

- **Mandatory checks (1-6, 19-20, 24):** Failure is Critical — blocks progress, Implementer must fix
- **Advisory checks (7-18, 21-23):** Failure is Warning — reported but does not block progress

## Fix Loop

Uses the shared fix loop from `spml:validation-pyramid`:
- Mandatory check fails → send to Implementer with specific file:line fix instructions
- Implementer fixes → re-run L0 review
- 5 consecutive failures → pause, notify user
- If fix modifies > 50 lines → rollback: re-run Spec Review + Code Quality Review + L0

## Checklist Reference

See `checklist.md` for the full conditional checklist.

## Integration

- **spml:ml-subagent-dev** — dispatches this as a review stage
- **spml:validation-pyramid** — L0 in the 2-level pyramid
- **spml:ml-runtime-validator** — next level after L0 passes
