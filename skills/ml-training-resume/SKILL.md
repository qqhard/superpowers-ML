---
name: ml-training-resume
description: Use when resuming from a Watchdog session — reads experiment context and diagnosis, autonomously decides which workflow stage to return to (fix code, adjust hyperparams, replan, or rebrainstorm)
---

# ML Training Resume

## Overview

Entry point for recovery or completion after a long-running ML task. Reads the experiment context file (written by handoff, updated by Watchdog) and autonomously decides what to do next.

**Core principle:** Decide from evidence, not assumptions. All information needed is in experiment-context.md and metrics.jsonl. The agent judges the rollback level itself.

## When to Use

- User has pasted a recovery-prompt.md (Watchdog found an issue)
- User has pasted a completion-prompt.md (training finished normally)

## Startup

1. **Read experiment-context.md** — the complete chain: experiment design, VP baseline, training config, code state, Watchdog diagnosis/summary
2. **Identify entry type:**
   - Has Diagnosis History with unresolved issue → recovery path
   - Watchdog Status is "completed" → completion path
   - Unclear → ask user

## Recovery Path

### Step 1: Understand the Diagnosis

Read the Watchdog's diagnosis in experiment-context.md:
- What symptom was observed?
- At what step/time?
- What metrics co-moved?
- What was the VP baseline for these metrics?

### Step 2: Analyze Root Cause

If needed, read metrics.jsonl directly for deeper analysis:
- Look at the metrics trajectory before/during/after the anomaly
- Compare with VP baseline ranges
- Check for patterns in the common anomaly table

### Step 3: Determine Rollback Level

Based on evidence, decide which workflow stage to return to:

| Evidence Pattern | Rollback To | Action |
|-----------------|-------------|--------|
| NaN/Inf, shape error, crash | Code fix | Fix bug → re-run VP → re-handoff |
| Loss not converging, gradients healthy, code correct | Hyperparameter adjustment | Adjust LR/batch/optimizer → possibly re-run VP L2 → re-handoff |
| Training works but wrong metrics tracked, wrong evaluation | Task spec fix | Back to ml-experiment-planning to adjust subtask spec |
| Fundamental approach doesn't work despite correct implementation | Hypothesis revision | Back to ml-brainstorming |
| Data quality issues (corrupted batches, distribution shift) | Data fix | Back to ml-data-preparation |

### Step 4: Execute

Based on the determined rollback level:

**Code fix:**
1. Checkout the git commit from experiment-context.md
2. Fix the identified issue
3. Re-run the failed VP checks to verify the fix
4. Invoke ml-training-handoff to generate new artifacts
5. New handoff appends Round N to experiment-context.md (preserving history)

**Hyperparameter adjustment:**
1. Identify which hyperparameters to change and why
2. If the change is significant, re-run VP L2 (overfit test) to verify
3. Invoke ml-training-handoff with updated training config
4. Optionally resume from the last known good checkpoint

**Task spec fix:**
1. Explain the issue to the user
2. Invoke ml-experiment-planning to revise the subtask
3. Re-execute the revised subtask through ml-subagent-dev

**Hypothesis revision:**
1. Explain why the current hypothesis appears invalid
2. Present evidence from the training run
3. Invoke ml-brainstorming for a new experiment design

**Data fix:**
1. Identify the data quality issue from training metrics
2. Invoke ml-data-preparation to fix and re-validate data
3. After data is fixed, re-handoff for training

### Step 5: Communicate

Always tell the user:
- What was diagnosed
- Which rollback level was chosen and why
- What happens next

If the agent is uncertain about the rollback level, present the evidence and ask the user to decide.

## Completion Path

### Step 1: Analyze Results

Read experiment-context.md Watchdog completion summary:
- Final metrics vs VP baseline
- Final metrics vs experiment hypothesis expectations
- Any anomalies that occurred during training

### Step 2: Deep Analysis (if needed)

Read metrics.jsonl for:
- Full training curve (loss trajectory)
- Metric stability in final phase
- Any concerning trends even if training "completed"

### Step 3: Enter ml-verification

Invoke the ml-verification flow:
- Validate all VP layers were originally passed
- Check that long-training results support the conclusion
- Compare hypothesis prediction vs actual outcome
- Compile experiment summary report
- Present decision options to user (finish / add subtasks / new brainstorm)

## Multi-Round Context

experiment-context.md may contain multiple rounds (Round 1, Round 2, ...) from previous recovery cycles. The agent should:
- Read all rounds to understand the full history
- Not repeat fixes that were already tried
- Recognize if the same problem recurs (may indicate a deeper issue)

## Red Flags

**Never:**
- Assume the problem without reading the evidence
- Skip directly to a fix without determining rollback level
- Ignore Diagnosis History from previous rounds
- Overwrite experiment-context.md (append Round N instead)

**Always:**
- Read experiment-context.md completely before deciding
- State the evidence and reasoning for the chosen rollback level
- Preserve experiment history across rounds
- Ask user when uncertain about the right level to rollback to

## Integration

- **spml:ml-watchdog** — Produces the prompts and context this skill consumes
- **spml:ml-training-handoff** — Re-invoked when fix requires another training run
- **spml:ml-verification** — Invoked on the completion path
- **spml:ml-brainstorming** — Invoked when hypothesis needs revision
- **spml:ml-experiment-planning** — Invoked when task decomposition needs revision
- **spml:ml-data-preparation** — Invoked when data quality is the issue
- **spml:ml-diagnostics** — May be invoked for deeper code-level diagnosis
