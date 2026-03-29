# Watchdog Async Evaluation Design

**Date:** 2026-03-30
**Status:** Draft
**Scope:** Watchdog skill + training-handoff experiment-context.md template

## Problem

Watchdog monitors long-running training but has no way to evaluate checkpoints during training. Evaluation either happens post-training or requires the user to manually trigger it. This means the user has no visibility into model quality during training, only training metrics (loss, MFU, etc.).

## Design Principle

Watchdog already wakes up periodically to check training health. When it discovers a new checkpoint, it spawns an evaluation subagent in the background — keeping the Watchdog's own context clean and training uninterrupted.

Evaluation frequency is naturally controlled by three factors:
- Checkpoint save interval (user-configured in training script)
- Watchdog polling interval (2-5 min)
- Whether the previous evaluation has completed (skip if busy)
- Whether a free GPU is available (skip if not)

No explicit evaluation frequency counter is needed.

## Changes

### 1. Monitoring Loop — Insert Step 3.5

Insert between existing step 3 (parse metrics) and step 4 (analyze metrics):

```
3.5. Check for new checkpoint:
     a. Scan checkpoint directory for newest checkpoint
     b. Compare with last_evaluated_checkpoint:
        - Same → skip
        - Newer AND all preconditions met → spawn eval subagent
        - Newer AND any precondition failed → skip, log reason
     c. Check if background eval subagent has returned:
        - Returned → read summary, append to experiment-context.md Evaluation History
        - Set eval_subagent_running=false, update last_evaluated_checkpoint
```

**Preconditions (all must be true):**
1. New checkpoint exists (newer than last_evaluated_checkpoint)
2. eval_subagent_running = false
3. eval_paused = false
4. Free GPU available (check via `nvidia-smi` — at least one GPU not fully occupied by training)

**When any precondition fails:**
- Log reason: "eval busy, skipping step=N" / "no free GPU, skipping step=N" / "eval paused"
- Continue monitoring loop normally

### 2. Eval Subagent

**Dispatch:** Use Agent tool with `run_in_background: true` so Watchdog continues its monitoring loop.

**Prompt template:**
```
Run evaluation on a training checkpoint.

Checkpoint path: <checkpoint_path>
Command: <eval_command from experiment-context.md>

Replace {checkpoint_path} in the command with the actual checkpoint path.
Replace {output_dir} with the evaluation output directory if present in the command.

Execute the command. When it completes, return a one-line summary in this format:
  step=N metric1=val1 metric2=val2 duration=Xm

Do NOT modify any training code or training data.
Do NOT modify checkpoints.
If the command fails, return: step=N status=FAILED error=<brief error description>
```

**On return:**
- Watchdog reads the summary from the Agent tool result
- Appends to experiment-context.md Evaluation History
- Sets eval_subagent_running=false
- Updates last_evaluated_checkpoint

### 3. experiment-context.md Template Changes

**Training Configuration section — add fields:**

```markdown
## Training Configuration
- ...existing fields...
- Eval command: <command with {checkpoint_path} placeholder, e.g., python eval.py --checkpoint={checkpoint_path}>
- Checkpoint directory: <path where checkpoints are saved>
```

`eval_command` is optional. If absent, Watchdog skips all evaluation (step 3.5 becomes a no-op).

**New section — Evaluation History:**

```markdown
## Evaluation History
(populated by Watchdog during training)

### Eval #1 — checkpoint step=5000
- Timestamp: 2026-03-30 14:20
- Summary: step=5000 loss=0.52 accuracy=0.78
- Duration: 3m12s

### Eval #2 — checkpoint step=10000
- Timestamp: 2026-03-30 15:45
- Summary: step=10000 loss=0.38 accuracy=0.84
- Duration: 3m08s
- Skipped checkpoints: step=7500 (eval busy), step=8500 (no free GPU)
```

### 4. User Commands — Evaluation Control

Add to existing user command handling in Watchdog:

| Command | Effect |
|---------|--------|
| "pause eval" | Set eval_paused=true, skip subsequent checkpoint evaluations |
| "resume eval" | Set eval_paused=false |
| "eval status" | Report: last evaluated checkpoint, eval subagent running?, skipped checkpoint count |

These are natural language commands — Watchdog as an LLM recognizes intent, not exact strings.

### 5. training-handoff Changes

training-handoff Step 3 "Expected" checklist — add:

```markdown
- [ ] If evaluation is part of the experiment: eval_command defined for Watchdog async evaluation
```

training-handoff Step 4 (write experiment-context.md) — add `Eval command` and `Checkpoint directory` to Training Configuration, add empty `Evaluation History` section.

## Watchdog State Variables

The following transient state is maintained in Watchdog's monitoring loop (not persisted to files):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| last_evaluated_checkpoint | string | null | Path/identifier of last evaluated checkpoint |
| eval_subagent_running | bool | false | Whether an eval subagent is currently running |
| eval_paused | bool | false | User has paused evaluation |

## Out of Scope

- Evaluation logic/scripts — the experiment provides these
- Evaluation-triggered training decisions (e.g., early stopping based on eval metrics) — future work
- Multi-GPU evaluation scheduling — Watchdog only checks for free GPU, doesn't manage allocation
