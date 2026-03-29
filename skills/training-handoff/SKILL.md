---
name: training-handoff
description: Use after VP passes when the task includes a long-running phase — verifies training script readiness, writes experiment context file, and generates Watchdog prompt for monitoring
---

# ML Training Handoff

## Overview

Bridge between VP validation (minute-level) and long-running execution (hours/days). Generates monitoring artifacts (experiment-context.md + watchdog-prompt.md) for a training script that has already been built and validated by the upstream flow.

**Core principle:** Do not rewrite VP-validated code. The training script was built by ml-subagent-dev, tested by VP L0/L1, and reviewed by spec+quality reviewers. Handoff's job is to verify it's production-ready and set up monitoring — not to modify it.

<HARD-GATE>
Do NOT hand off without:
1. All enabled VP layers passed
2. Training script exists and was validated by VP (L1 ran it successfully)
3. experiment-context.md written with VP baseline
</HARD-GATE>

## When to Use

- All VP checks passed for a subtask or experiment
- The task requires a long-running execution phase (training, full-scale data processing, large-scale evaluation)
- The task is NOT something that completes in minutes (those go directly to verification)

## Checklist

1. **Verify VP completion** — all enabled layers passed with actual numbers
2. **Verify directory layout** — experiment artifacts are co-located
3. **Verify training script readiness** — check existing script against production requirements
4. **Write experiment-context.md** — full context for downstream sessions
5. **Write watchdog-prompt.md** — prompt for Watchdog session
6. **Present launch instructions** — how to start the Watchdog session

## Step 1: Verify VP Completion

Confirm all VP layers that were enabled in the brainstorm design doc have passed. Record the actual metrics as VP baseline — the Watchdog will use these as reference.

## Step 2: Verify Directory Layout

Check that the experiment follows the co-located layout convention from `spml:experiment-planning`:

- Training script, tests, and outputs are under the same experiment directory
- Handoff artifacts (experiment-context.md, watchdog-prompt.md) will be written to that same directory

**Not a hard gate** — the user may have reasons for a different layout. If artifacts would be written outside the training script's directory, flag this to the user and ask whether to proceed or reorganize.

## Step 3: Verify Training Script Readiness

The training script already exists — it was implemented during ml-subagent-dev and validated by VP. Do NOT rewrite it. Instead, verify it meets production requirements:

**Required (block handoff if missing):**
- [ ] Zero agent dependency — runs standalone (`python train.py` or `bash run.sh`)
- [ ] Log output to file — Watchdog needs something to read
- [ ] Fixed seeds — for reproducibility

**Expected (flag gap if missing, ask user whether to quick-fix or proceed):**
- [ ] Console output uses tqdm (preferred) or controlled print, minute-level, carrying key metrics
- [ ] Detailed metrics written to file: loss, gradient norm, learning rate, step time
- [ ] MFU in file log (needed for efficiency monitoring)
- [ ] Checkpoint save with configurable interval
- [ ] If evaluation is part of the experiment: eval_command defined for Watchdog async evaluation
- [ ] Resumable from checkpoint
- [ ] If evaluation is part of the experiment: a distinct evaluation capability, not only a final-epoch block
- [ ] If evaluation is part of the experiment: both evaluation entry modes are available
  - checkpoint-based evaluation
  - in-memory evaluation during training
- [ ] Long-running evaluation has explicit phase messages and a dedicated progress bar
- [ ] Evaluation emits both result summary and efficiency/latency summary
- [ ] Evaluation reports mode-aware status:
  - checkpoint mode reports checkpoint path/load behavior
  - in-training mode reports that evaluation used in-memory state

**If gaps are found:**
- Do NOT silently rewrite the script
- List the gaps explicitly to the user
- Ask: "Fix these before handoff, or proceed as-is?"
- If fixing, make minimal targeted changes (not a rewrite) and re-run the affected VP layer to verify nothing broke

**No smoke test needed.** VP L1 already ran the script for a meaningful number of steps (far more thorough than a 2-step smoke test). If VP L1 passed, the script works.

### Upstream: Production Script Requirements

These requirements should be part of the experiment plan so ml-subagent-dev implements them from the start. If you find yourself frequently patching scripts at handoff, the fix is in `spml:experiment-planning`, not here. Key requirements to include in plans:

- Console output: tqdm (preferred) or controlled print, minute-level frequency, carrying key metrics (loss, lr, etc.)
- File output: detailed metrics written to file (loss, grad_norm, lr, mfu, memory, step_time), for Agent exploration
- Checkpoint: periodic save with configurable interval, resume support
- Step-based evaluation cadence
- `full validation` as the default evaluation scope unless explicitly overridden
- One shared evaluator core across checkpoint-based and in-memory evaluation
- Mode-aware evaluation errors and observability from the start

### Evaluation Readiness Checks

When the experiment includes evaluation, verify these explicitly during handoff:

- in-training evaluation is configured to fire at the planned step cadence
- checkpoint-based evaluation exists as an independent entrypoint
- long-running evaluation does not go silent: phase-start message, progress updates, phase-end message
- evaluation summary includes metrics plus efficiency details such as total duration, throughput, and checkpoint load latency where applicable
- evaluation errors surface with mode-aware context at the boundary:
  - checkpoint missing or unreadable
  - checkpoint restore failure
  - empty or misconfigured validation dataloader
  - metric aggregation failure
  - non-finite metrics
  - stalled evaluation / long silent output gaps

Treat cadence firing, evaluation mode reporting, progress visibility, and mode-aware boundary errors as runtime readiness checks. They are not stylistic suggestions.

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
- Pipeline stages: [all 6 passed / which failed]
- [Architecture-specific metrics]

## Training Configuration
- Script: [experiment-dir]/train.py (or actual script name)
- Launch command: [exact command to run training, including all arguments]
- Log file: [experiment-dir]/outputs/train.log
- Checkpoint directory: [experiment-dir]/outputs/
- Expected total steps: [N]
- Estimated duration: [hours]
- Key hyperparameters: [lr, batch_size, etc.]
- Eval command: [command with {checkpoint_path} placeholder, e.g., python eval.py --checkpoint={checkpoint_path}; leave empty if no evaluation]

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

## Evaluation History
(populated by Watchdog during training)
```

## Step 5: Write watchdog-prompt.md

```markdown
I need you to act as a Watchdog Agent, monitoring and shepherding a long-running ML task.

## Setup
1. Read `[experiment-dir]/experiment-context.md` for full experiment context, VP baseline, and watchdog mode
2. Locate the training log at `[experiment-dir]/outputs/train.log`
3. Locate the training script: `[exact launch command]`

## Your Behavior
Use the spml:watchdog skill. It will guide you through:
- Operating mode selection (Monitor/Guardian/Autonomous — preset in experiment-context.md, you can ask to override)
- Launching the training script
- Monitoring the training log for anomalies
- Taking action based on the operating mode and problem classification (Tier 1/2/3)
- Recording all interventions in experiment-context.md
- Notifying you when training finishes or encounters issues
```

## Step 6: Present Launch Instructions

Show the artifact summary, then **print the full watchdog-prompt.md content directly in the conversation** so the user can copy-paste it into a new session without opening any files.

Format:

```
Handoff complete. All artifacts generated:
- Training script: [experiment-dir]/train.py
- Log file: [experiment-dir]/outputs/train.log (human-readable, one line per step)
- Experiment context: [experiment-dir]/experiment-context.md
- Watchdog prompt: [experiment-dir]/watchdog-prompt.md
- Watchdog mode: [mode] (configurable in experiment-context.md)

To start — copy the prompt below into a new agent session:
```

Then output the **full content** of the generated watchdog-prompt.md in a fenced code block so it is directly copy-pasteable. Do NOT just say "paste the contents of watchdog-prompt.md" — the user should never need to open that file manually.

## Integration

- **spml:ml-subagent-dev** — Triggers handoff after VP passes (when long-running phase needed)
- **spml:watchdog** — The Watchdog prompt references this skill's behavior
- **spml:verification** — Skipped at handoff; entered later via re-entry on experiment directory
