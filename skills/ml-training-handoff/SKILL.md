---
name: ml-training-handoff
description: Use after VP passes when the task includes a long-running phase — generates production-ready training script, structured logging, experiment context file, and Watchdog prompt for monitoring
---

# ML Training Handoff

## Overview

Bridge between VP validation (minute-level) and long-running execution (hours/days). Generates everything needed for the user to run training independently and optionally monitor it with a Watchdog Agent session.

**Core principle:** Training scripts are core code — independently deployable to production, zero agent dependency. The monitoring protocol (prompt + context file + JSONL logs) is framework-agnostic.

<HARD-GATE>
Do NOT hand off without:
1. All enabled VP layers passed
2. Training script tested (at least 1-step smoke test)
3. JSONL logging verified (at least 1 line written)
4. experiment-context.md written with VP baseline
</HARD-GATE>

## When to Use

- All VP checks passed for a subtask or experiment
- The task requires a long-running execution phase (training, full-scale data processing, large-scale evaluation)
- The task is NOT something that completes in minutes (those go directly to ml-verification)

## Checklist

1. **Verify VP completion** — all enabled layers passed with actual numbers
2. **Generate training script** — production-ready, with dual output logging
3. **Smoke test** — run 1-2 steps to verify script works and JSONL is written
4. **Write experiment-context.md** — full context for downstream sessions
5. **Write watchdog-prompt.md** — short prompt for Watchdog session
6. **Present user options** — separated or combined execution

## Step 1: Verify VP Completion

Confirm all VP layers that were enabled in the brainstorm design doc have passed. Record the actual metrics as VP baseline — the Watchdog will use these as reference.

## Step 2: Generate Training Script

The training script is core code. Requirements:

- **Zero agent dependency** — runs with `bash run_training.sh` or `python train.py`
- **Dual output logging:**
  - Terminal: tqdm progress bar (one line, continuously updating)
  - File: JSONL with all tracked metrics per step
- **Checkpoint support** — periodic saves, configurable interval
- **Resumable** — can restart from a checkpoint
- **Fixed seeds** — for reproducibility

**tqdm progress bar (terminal):**
```python
from tqdm import tqdm

pbar = tqdm(range(total_steps), desc="Training")
for step in pbar:
    # ... training step ...
    pbar.set_postfix(loss=f"{loss:.3f}", MFU=f"{mfu:.0%}")
```

**JSONL logging (file):**
```python
import json

def log_metrics(log_file, step, metrics):
    metrics["step"] = step
    metrics["timestamp"] = datetime.now().isoformat()
    with open(log_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")
```

The JSONL should include all metrics that VP L1 confirmed are worth tracking: loss, gradient norm, learning rate, MFU, memory usage, and any architecture-specific metrics (attention entropy, MoE balance, etc.).

## Step 3: Smoke Test

Run the training script for 1-2 steps to verify:
- Script starts without errors
- tqdm progress bar appears
- JSONL file is created with at least 1 valid line
- Checkpoint directory is created

```bash
python train.py --max-steps 2 --log-file logs/smoke_test.jsonl
# Verify JSONL
python -c "import json; [json.loads(l) for l in open('logs/smoke_test.jsonl')]"
```

## Step 4: Write experiment-context.md

```markdown
# Experiment Context: [name]

## Experiment Design
- Hypothesis: [from brainstorm]
- Independent variable: [what changes]
- Dependent variable: [what to measure]
- Control variable: [what stays the same]
- Validation scope: [which VP layers were enabled]

## VP Baseline
[Actual metrics from VP runs — these are the Watchdog's reference]
- MFU: [value]
- Gradient norm range: [min-max observed]
- Initial loss: [value]
- Overfit test result: [if L2 was run]
- [Architecture-specific metrics]

## Training Configuration
- Script: [path to run_training.sh or train.py command]
- Log file: [path to metrics.jsonl]
- Checkpoint directory: [path]
- Expected total steps: [N]
- Estimated duration: [hours]
- Key hyperparameters: [lr, batch_size, etc.]

## Code State
- Git commit: [hash]
- Branch: [name]
- Key files: [list of main files]

## Watchdog Status
- Status: not started

## Diagnosis History
(empty)
```

## Step 5: Write watchdog-prompt.md

```markdown
I need you to act as a Watchdog Agent, monitoring a long-running ML task.

Your job is to OBSERVE and DIAGNOSE, never intervene.

## Setup
1. Read `[path]/experiment-context.md` for full experiment context and VP baseline
2. Read `[path]/logs/metrics.jsonl` for training metrics

## Your Behavior
- Periodically read the latest entries from the JSONL log
- Compare metrics against the VP baseline in experiment-context.md
- Detect anomalies: spikes, NaN, stagnation, sudden trend changes
- Adaptive frequency: check often at start (first 10% of steps), less in steady state, more near completion
- You can adjust frequency if the user asks

## Output
- When monitoring in combined mode (you launched the training): report progress periodically
- When monitoring in separated mode (training runs elsewhere): stay silent when normal, speak only on anomaly
- On anomaly: describe what you see, write diagnosis to experiment-context.md, produce recovery-prompt.md
- On normal completion: summarize results, write to experiment-context.md, produce completion-prompt.md

## Anomaly Diagnosis
When you detect a problem:
1. Collect evidence from JSONL (what changed, when, which metrics co-moved)
2. Compare against VP baseline (is this outside the range we saw in quick validation?)
3. Write your diagnosis to the "Diagnosis History" section of experiment-context.md
4. Create recovery-prompt.md with a short prompt that tells the next agent to read experiment-context.md

## Completion
When training finishes normally:
1. Summarize final metrics
2. Compare against VP baseline and experiment hypothesis
3. Update experiment-context.md
4. Create completion-prompt.md with a short prompt that tells the next agent to read experiment-context.md and run ml-verification

## Boundaries
- DO: read logs, analyze trends, write reports, produce prompts, notify user
- DO NOT: stop training, modify code, adjust hyperparameters, rollback checkpoints
```

## Step 6: Present User Options

```
Handoff complete. All artifacts generated:
- Training script: [path]
- Metrics log: [path] (JSONL)
- Experiment context: [path]/experiment-context.md
- Watchdog prompt: [path]/watchdog-prompt.md

Two execution options:

Option A: Separated execution
  1. Start training: [exact command]
  2. Open a new agent session and paste the contents of watchdog-prompt.md

Option B: Combined execution
  1. Open a new agent session and paste the contents of watchdog-prompt.md
  2. Ask the Watchdog to launch training and begin monitoring
```

## Integration

- **spml:ml-subagent-dev** — Triggers handoff after VP passes (when long-running phase needed)
- **spml:ml-watchdog** — The Watchdog prompt references this skill's behavior
- **spml:ml-verification** — Skipped at handoff; entered later via resume
