---
name: training-handoff
description: Use after VP passes when the task includes a long-running phase — generates production-ready training script, human-readable logging, experiment context file, and Watchdog prompt for monitoring
---

# ML Training Handoff

## Overview

Bridge between VP validation (minute-level) and long-running execution (hours/days). Generates everything needed to run training and monitor it with a Watchdog Agent session.

**Core principle:** Training scripts are core code — independently deployable to production, zero agent dependency. The monitoring protocol (prompt + context file + log file) is framework-agnostic.

<HARD-GATE>
Do NOT hand off without:
1. All enabled VP layers passed
2. Training script tested (at least 1-step smoke test)
3. Log output verified (at least 1 line written to log file)
4. experiment-context.md written with VP baseline
</HARD-GATE>

## When to Use

- All VP checks passed for a subtask or experiment
- The task requires a long-running execution phase (training, full-scale data processing, large-scale evaluation)
- The task is NOT something that completes in minutes (those go directly to verification)

## Checklist

1. **Verify VP completion** — all enabled layers passed with actual numbers
2. **Generate training script** — production-ready, with human-readable logging
3. **Smoke test** — run 1-2 steps to verify script works and log output is written
4. **Write experiment-context.md** — full context for downstream sessions
5. **Write watchdog-prompt.md** — prompt for Watchdog session
6. **Present launch instructions** — how to start the Watchdog session

## Step 1: Verify VP Completion

Confirm all VP layers that were enabled in the brainstorm design doc have passed. Record the actual metrics as VP baseline — the Watchdog will use these as reference.

## Step 2: Generate Training Script

The training script is core code. Requirements:

- **Zero agent dependency** — runs with `bash run_training.sh` or `python train.py`
- **Human-readable log output:**
  - Terminal: tqdm progress bar or similar (for human monitoring)
  - File: one line per step/epoch with key metrics (for Watchdog monitoring)
- **Checkpoint support** — periodic saves, configurable interval
- **Resumable** — can restart from a checkpoint and continue from where it left off
- **Fixed seeds** — for reproducibility

**Terminal output (tqdm):**
```python
from tqdm import tqdm

pbar = tqdm(range(total_steps), desc="Training")
for step in pbar:
    # ... training step ...
    pbar.set_postfix(loss=f"{loss:.3f}", MFU=f"{mfu:.0%}")
```

**Log file output (human-readable, one line per step):**
```python
import logging

logger = logging.getLogger("training")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)

# After each step:
logger.info(f"step={step} loss={loss:.4f} lr={lr:.6f} grad_norm={grad_norm:.4f} mfu={mfu:.4f} mem_mb={mem_mb}")
```

The log file should include all metrics that VP L1 confirmed are worth tracking: loss, gradient norm, learning rate, MFU, memory usage, and any architecture-specific metrics (attention entropy, MoE balance, etc.).

The exact format is flexible (key=value, tabular, or any readable layout). The Watchdog is an LLM and parses any consistent text format.

## Step 3: Smoke Test

Run the training script for 1-2 steps to verify:
- Script starts without errors
- tqdm progress bar appears
- Log file is created with at least 1 line containing metrics
- Checkpoint directory is created

```bash
python train.py --max-steps 2 --log-file logs/training.log
# Verify log file has content
cat logs/training.log
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
- Launch command: [exact command to run training, including all arguments]
- Log file: [path to training.log]
- Checkpoint directory: [path]
- Expected total steps: [N]
- Estimated duration: [hours]
- Key hyperparameters: [lr, batch_size, etc.]

## Watchdog Configuration
- watchdog_mode: guardian
  (Options: monitor | guardian | autonomous)
  (monitor = report only; guardian = auto-restart + auto-fix simple issues + report complex; autonomous = handle everything including complex issues via sub-agent)

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
I need you to act as a Watchdog Agent, monitoring and shepherding a long-running ML task.

## Setup
1. Read `[path]/experiment-context.md` for full experiment context, VP baseline, and watchdog mode
2. Locate the training log at `[path]/logs/training.log`
3. Locate the training script: `[exact launch command]`

## Your Behavior
Use the spml:watchdog skill. It will guide you through:
- Operating mode selection (Monitor/Guardian/Autonomous — preset in experiment-context.md, you can ask to override)
- Launching the training script
- Monitoring the training log for anomalies
- Taking action based on the operating mode and problem classification (Tier 1/2/3)
- Recording all interventions in experiment-context.md
- Producing completion-prompt.md when training finishes
```

## Step 6: Present Launch Instructions

```
Handoff complete. All artifacts generated:
- Training script: [path]
- Log file: [path] (human-readable, one line per step)
- Experiment context: [path]/experiment-context.md
- Watchdog prompt: [path]/watchdog-prompt.md
- Watchdog mode: [mode] (configurable in experiment-context.md)

To start:
  1. Open a new agent session
  2. Paste the contents of watchdog-prompt.md
  3. The Watchdog will launch training and begin monitoring
```

## Integration

- **spml:subagent-dev** — Triggers handoff after VP passes (when long-running phase needed)
- **spml:watchdog** — The Watchdog prompt references this skill's behavior
- **spml:verification** — Skipped at handoff; entered later via resume
