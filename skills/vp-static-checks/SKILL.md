---
name: ml-code-reviewer
description: Use when reviewing ML code for static correctness — dispatched after Spec Review and Code Quality Review in the subagent-dev workflow
---

# L0: ML Static Analysis (ML Code Reviewer)

## Overview

A specialized code-reviewer subagent that checks ML-specific static correctness. Runs after standard Spec Review and Code Quality Review pass. Catches configuration errors, device mismatches, precision problems, and optimization oversights — issues that ML agents commonly introduce and that don't require runtime to detect.

**This is a RIGID skill.** Follow the checklist exactly. Don't skip applicable checks.

## When to Use

- Automatically dispatched by the orchestrator in `spml:subagent-dev` after code quality review passes
- Only for tasks that involve ML code (model, training loop, data pipeline, optimizer config)
- Skip for pure infrastructure tasks (CI, docs, config files)

## How It Works

1. Orchestrator dispatches `ml-code-reviewer` agent (defined in `agents/ml-code-reviewer.md`)
2. Agent reads all changed files
3. Agent evaluates each checklist item's applicability condition
4. For applicable items: verify the code meets the requirement
5. Report findings using Critical/Important/Suggestion severity levels

## Severity Tiers

- **Mandatory checks (1-6):** Failure is Critical — blocks progress, Implementer must fix
- **Advisory checks (7-18):** Failure is Warning — reported but does not block progress

## Fix Loop

Uses the shared fix loop from `spml:validation-pyramid`:
- Mandatory check fails → send to Implementer with specific file:line fix instructions
- Implementer fixes → re-run L0 review
- 5 consecutive failures → pause, notify user
- If fix modifies > 50 lines → rollback: re-run Spec Review + Code Quality Review + L0

## Checklist Reference

See `checklist.md` for the full conditional checklist.

## Integration

- **spml:subagent-dev** — dispatches this as a review stage
- **spml:validation-pyramid** — L0 in the 3-level pyramid
- **spml:ml-runtime-validator** — next level after L0 passes
