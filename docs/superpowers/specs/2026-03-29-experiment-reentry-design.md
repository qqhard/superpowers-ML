> **Note:** References to L2/ml-e2e-validator in this doc are stale — L2 has been merged into L1. See `docs/superpowers/specs/2026-03-29-vp-l1-l2-merge-design.md`.

# Experiment Re-entry: Maintainable Experiment Directories

## Problem

When an experiment already has design docs, plans, and code, starting a new session and requesting changes causes the AI to create brand new design/plan files instead of reading and revising the existing ones. The old files become stale, the new files have no connection to the old, and the experiment becomes unmaintainable.

**Root cause:** All SPML skills assume a fresh start. No skill checks for existing artifacts or offers a "revision" mode. The entry routing (`using-superpowers-ml`) doesn't detect experiment directory state.

## Core Principle

**Design, plan, and code must always be consistent.** Changes propagate top-down:

```
design (source of truth) → plan (derived) → code (implementation)
```

- Any change starts from design
- Design change → check which plan subtasks are affected → update plan → re-execute affected subtasks
- Direct code changes without design/plan updates are forbidden

## Design

### Change 1: Directory State Detection in `using-superpowers-ml`

Add experiment directory scanning before routing to any skill.

**Detection logic:**

```
Scan <experiment-dir>/plans/:
  - *-design.md exists? → has_design = true
  - non-design *.md exists? → has_plan = true

Scan <experiment-dir>/:
  - *.py exists? → has_code = true

State matrix:
┌────────────┬──────────┬──────────┬──────────────────────────┐
│ has_design │ has_plan │ has_code │ Action                   │
├────────────┼──────────┼──────────┼──────────────────────────┤
│ false      │ false    │ false    │ New: ml-brainstorming    │
│ true       │ false    │ false    │ Read design → planning   │
│ true       │ true     │ false    │ Read plan → subagent-dev │
│ true       │ true     │ true     │ Read design → revision   │
└────────────┴──────────┴──────────┴──────────────────────────┘
```

Last row is the key case: full experiment exists, user wants changes → enter ml-brainstorming in revision mode.

**HARD-GATE:** When existing artifacts are detected, ALL design docs and plans must be read BEFORE invoking any skill. No skill may create new design/plan files when existing ones can be revised.

### Change 2: `ml-brainstorming` Revision Mode

When existing design doc content is passed by the orchestrator, brainstorming enters revision mode.

**What changes:**
- Skip "Collecting ML Context" questions that already have answers in the existing design (hypothesis, variables, dataset, etc.)
- Present existing design summary first: "Current design: [1-3 sentence summary]. What do you want to change?"
- Only ask questions about the delta — what's changing and why
- Edit the existing design doc in place (not create new file)
- Commit: "experiment: revise design — [what changed]"

**What stays the same:**
- User approval required before proceeding
- Spec self-review still runs
- Transitions to experiment-planning (also in revision mode)

**Impact tracking:** After revision, append a section to the design doc:

```markdown
## Impact on Plan
- Subtask N: [needs update because X changed]
- Subtask M: [unaffected]
- New subtask needed: [description]
```

### Change 3: `experiment-planning` Revision Mode

When existing plan content AND a revised design with "## Impact on Plan" section are passed, planning enters revision mode.

**Flow:**
1. Read existing plan fully
2. Read design's Impact section — which subtasks are affected
3. For affected subtasks: rewrite steps to match revised design, preserve numbering
4. For new subtasks: append to end (Task N+1, N+2, ...)
5. For removed subtasks: mark as "REMOVED: [reason]" (don't delete)
6. Edit existing plan file in place
7. Commit: "experiment: revise plan — [what changed]"

**Unchanged subtasks that already passed VP keep their results.** Mark in plan:
```
- [x] Task 1: ... (unchanged, VP passed)
- [ ] Task 2: ... (REVISED — needs re-execution)
- [ ] Task 5: ... (NEW)
```

Plan Gate still applies. Self-review still runs.

### Change 4: `ml-subagent-dev` Revision Mode Adaptation

When plan has revision markers:

- `[x]` = unchanged, skip (VP results preserved)
- `[ ] REVISED` = needs re-execution on existing code
  - Implementer gets old code path as context
  - Modifies existing code (not from scratch)
  - VP must fully re-run (old VP results voided)
  - Spec Review + Quality Review must re-run
- `[ ] NEW` = normal fresh flow

Completion Gate unchanged — all re-executed subtasks must pass full gate.

### Change 5: Delete `training-resume`

**Current `training-resume` functionality:**
- Read experiment-context.md → determine rollback level
- Options: brainstorm / plan / code fix / hyperparams / data fix

**Replaced by unified re-entry:**
- All rollback paths are handled by the directory state detection + revision modes
- Brainstorm rollback → user enters experiment dir, says "change X" → revision mode
- Plan rollback → same, design unchanged, directly revise plan
- Code fix → design → plan → re-execute affected subtasks
- Hyperparams → design change, same flow

**What's preserved:**
- `experiment-context.md` as training record (VP baseline, training config)
- Watchdog still runs independently
- Watchdog no longer calls training-resume; instead tells user: "Training issue: [description]. Fix in a new session on the experiment directory."

**File operations:**
- Delete `skills/training-resume/SKILL.md`
- Remove references from `using-superpowers-ml`
- Remove references from `ml-subagent-dev` Integration section
- Update `watchdog` to replace training-resume references with unified re-entry instructions

### Change 6: Delete `spml:executing-plans`

`spml:executing-plans` is a copy of `superpowers:executing-plans` with no ML-specific additions. It should be deleted. Users should use `superpowers:executing-plans` directly.

**File operations:**
- Delete `skills/executing-plans/SKILL.md`
- Remove references from `using-superpowers-ml`
- Update any skill that references `spml:executing-plans` to point to `superpowers:executing-plans`

### Change 7: Remove `recovery-prompt.md` and `completion-prompt.md`

These artifacts are redundant given the unified re-entry mechanism. The experiment directory itself is the state — no intermediate prompt files needed.

**What changes:**
- `watchdog/SKILL.md`: remove logic that generates `recovery-prompt.md` and `completion-prompt.md`
- `watchdog/SKILL.md`: when training fails or completes, instead output: "Training [completed/failed]: [description]. Start a new session on the experiment directory to continue."
- `training-handoff/SKILL.md`: remove references to these artifacts if any

### Change 8: VP Background Execution Timeout Protection

During VP execution (L1 runtime validator, L2 E2E validator), code may be sent to background execution and hang. The main session has no mechanism to detect this, causing the entire flow to stall.

**Fix:** When VP dispatches a background execution, the orchestrator MUST start a periodic liveness check.

**Timeout per VP layer:**

| Layer | Total Timeout | Check Interval |
|-------|--------------|----------------|
| L0: Static Checks | 2 minutes | N/A (not background) |
| L1: Runtime Validator | 10 minutes | 30 seconds |
| L2: E2E Validator | 15 minutes | 30 seconds |

**Liveness check behavior:**
1. After dispatching code to background, start a check loop at the specified interval
2. Each check: is the process still running? Has it exceeded the total timeout?
3. If timeout exceeded → kill the background process → report timeout error → enter fix loop (same as VP failure)
4. If process completes within timeout → read output → continue normal VP flow

**Where to add this:**
- `skills/ml-runtime-validator/SKILL.md` — add timeout + liveness check requirement for L1
- `skills/ml-e2e-validator/SKILL.md` — add timeout + liveness check requirement for L2
- `skills/ml-subagent-dev/SKILL.md` — document that VP layers have timeout protection; timeout failure = VP failure → enter fix loop

## Scope

**Modified files:**
- `skills/using-superpowers-ml/SKILL.md` — add directory state detection + routing
- `skills/ml-brainstorming/SKILL.md` — add revision mode
- `skills/experiment-planning/SKILL.md` — add revision mode
- `skills/ml-subagent-dev/SKILL.md` — add revision mode adaptation + VP timeout documentation
- `skills/watchdog/SKILL.md` — replace training-resume references, remove recovery/completion prompt generation
- `skills/ml-runtime-validator/SKILL.md` — add L1 timeout + liveness check
- `skills/ml-e2e-validator/SKILL.md` — add L2 timeout + liveness check
- `skills/training-handoff/SKILL.md` — remove recovery/completion prompt references (if any)

**Deleted files:**
- `skills/training-resume/SKILL.md`
- `skills/executing-plans/SKILL.md`

**Not changed:**
- `skills/ml-static-checks/SKILL.md` — L0 is synchronous, no timeout needed
- `skills/validation-pyramid/SKILL.md` — unchanged (orchestration spec only)
- `skills/verification/SKILL.md` — unchanged
- `skills/diagnostics/SKILL.md` — unchanged
