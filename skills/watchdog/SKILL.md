---
name: watchdog
description: Use when monitoring a long-running ML task — read-only observation of training/data processing/evaluation with adaptive frequency, anomaly detection, and diagnosis reporting
---

# ML Watchdog

## Overview

Read-only monitoring agent for long-running ML tasks. Periodically reads structured logs, compares against VP baseline, detects anomalies through trend analysis and pattern recognition. Produces recovery or completion prompts for the next session.

**Core principle:** Watch, don't act. The Watchdog observes and diagnoses but never intervenes in the running process. When issues are found, it packages context for a human or recovery agent to decide next steps.

**This skill is framework-agnostic.** The Watchdog prompt can be executed by any LLM agent (Claude Code, Codex, Cursor, etc.).

<HARD-GATE>
Do NOT take any action that modifies the running process, codebase, or training configuration.
The ONLY files you write are: experiment-context.md (updates), recovery-prompt.md, completion-prompt.md.
</HARD-GATE>

## When to Use

- User has pasted a Watchdog prompt from training-handoff
- A long-running ML task is running or about to start (training, data processing, evaluation)

## Startup

1. **Read experiment-context.md** — understand experiment design, VP baseline, expected behavior, training config
2. **Verify log file exists** — check that the JSONL metrics file is at the expected path
3. **Determine execution mode:**
   - If training is already running (log file has entries) → separated mode (silent monitoring)
   - If training hasn't started → ask user: "Should I launch the training script from this session?"
     - Yes → combined mode (launch training, report progress + monitor)
     - No → separated mode (wait for log file to appear)
4. **Enter monitoring loop**

## Monitoring Loop

```
loop {
    1. Read latest entries from JSONL log (tail, not full re-read)
    2. Analyze:
       a. Sanity: NaN, Inf, negative loss, zero gradient
       b. Baseline comparison: current metrics vs VP baseline ranges
       c. Trend: is loss decreasing? gradient norm stable? MFU consistent?
       d. Anomaly patterns: spike, plateau, divergence, sudden shift
    3. Classify:
       - NORMAL: metrics within expected ranges, healthy trends
       - MILD_ANOMALY: slight deviation, worth watching
       - SEVERE_ANOMALY: clear problem requiring attention
       - COMPLETE: training finished (final step reached or early stop triggered)
       - STALE: no new log entries for extended period (process may have died)
    4. Act on classification:
       - NORMAL → output (mode-dependent), continue
       - MILD_ANOMALY → log observation, increase frequency, continue
       - SEVERE_ANOMALY → enter diagnosis mode
       - COMPLETE → enter completion mode
       - STALE → notify user, ask to check process
    5. Wait (adaptive interval)
}
```

## Adaptive Frequency

| Training Phase | Check Interval | Rationale |
|----------------|---------------|-----------|
| First 10% of steps | 1-2 minutes | Catch startup issues early |
| 10%-90% of steps | 5-10 minutes | Steady-state, less frequent |
| Last 10% of steps | 2-3 minutes | Watch final convergence |
| After mild anomaly | 1 minute | Close observation until resolved |
| User override | As requested | "Check now" or "every N minutes" |

Calculate training phase from: current step / total steps (from experiment-context.md).

## Output Behavior

### Separated Mode (training runs in another terminal)

- **Normal:** Silent. No output.
- **Mild anomaly:** Brief note, continue watching.
- **Severe anomaly:** Detailed diagnosis + recovery prompt.
- **Complete:** Summary + completion prompt.

### Combined Mode (training launched from this session)

- **Normal:** Periodic progress report:
  ```
  [14:10] step 2000/10000 (20%) — loss=0.52, MFU=45%, grad_norm=1.8 [estimated 3:20 remaining]
  ```
- **Mild anomaly:** Progress report with warning flag.
- **Severe anomaly:** Detailed diagnosis + recovery prompt.
- **Complete:** Summary + completion prompt.

## Diagnosis Mode

Triggered by SEVERE_ANOMALY classification.

### Step 1: Collect Evidence

From the JSONL log, gather:
- Metrics at the anomaly point and 10-20 steps before
- Which metrics changed simultaneously
- Whether the change was sudden or gradual
- Step number and timestamp of onset

### Step 2: Contextualize

Compare against experiment-context.md:
- Is this metric outside the VP baseline range?
- Does the training config explain this? (e.g., LR schedule transition)
- Has a similar issue been recorded in Diagnosis History? (recurring problem)

### Step 3: Write Diagnosis

Update experiment-context.md, appending to Diagnosis History:

```markdown
### Issue #N — step [step_number]
- Symptom: [what happened, with numbers]
- Context: [what was expected at this training phase]
- Co-occurring signals: [other metrics that moved]
- Duration: [how long the anomaly lasted before Watchdog intervened]
- Checkpoint reference: [nearest checkpoint before/after anomaly]
- Recovery prompt: recovery-prompt.md
```

### Step 4: Produce recovery-prompt.md

```markdown
A long-running ML task has encountered an issue. Please diagnose and decide next steps.

Read `[path]/experiment-context.md` for the full context including:
- Original experiment design and hypothesis
- VP baseline metrics
- Watchdog diagnosis of the issue

Read `[path]/logs/metrics.jsonl` if you need to analyze the raw training metrics.

Your job: determine what went wrong, decide which workflow stage to return to, and fix it.
```

### Step 5: Notify User

```
⚠ Anomaly detected at step [N].
Diagnosis written to experiment-context.md.
To investigate: open a new agent session and paste the contents of recovery-prompt.md.
```

## Completion Mode

Triggered when training reaches final step or early stop condition.

### Step 1: Final Summary

Read the full JSONL log and summarize:
- Final metric values
- Training trajectory (was convergence smooth?)
- Total training time
- Any anomalies that occurred and resolved during training

### Step 2: Compare Against Expectations

- Final loss vs what was expected from VP L2 overfit test extrapolation
- MFU consistency vs VP L0 baseline
- Any metric drift from VP baseline

### Step 3: Update experiment-context.md

```markdown
## Watchdog Status
- Status: completed
- Total steps: [N]
- Total time: [duration]
- Final metrics: loss=[val], MFU=[val], ...
- Anomalies during training: [count, brief summary]
```

### Step 4: Produce completion-prompt.md

```markdown
A long-running ML task has completed successfully. Please analyze results and conclude the experiment.

Read `[path]/experiment-context.md` for the full context including:
- Original experiment design and hypothesis
- VP baseline metrics
- Watchdog monitoring summary
- Final training metrics

Your job: compare results against the hypothesis, run verification, and present conclusions to the user.
```

### Step 5: Notify User

```
✓ Training complete. [total_steps] steps in [duration].
Final loss: [val], MFU: [val].
To analyze results: open a new agent session and paste the contents of completion-prompt.md.
```

## Common Anomaly Patterns

| Pattern | Indicators | Typical Cause |
|---------|-----------|---------------|
| Loss spike | Sudden loss increase > 3x recent average | Bad batch, LR too high, numerical instability |
| Loss plateau | Loss unchanged for > 5% of total steps | LR too low, dead neurons, data exhaustion |
| Gradient explosion | grad_norm > 10x VP baseline | LR too high, loss function issue |
| Gradient vanishing | grad_norm < 0.01x VP baseline | Vanishing gradients, architecture issue |
| NaN/Inf | Any NaN or Inf in metrics | Numerical instability, log(0), division by zero |
| MFU drop | MFU drops > 20% from VP baseline | Thermal throttle, I/O bottleneck, memory pressure |
| Stale log | No new entries for > 3x expected step time | Process crashed, OOM, disk full |

## Red Flags

**Never:**
- Kill or restart the training process
- Modify any code files
- Adjust hyperparameters or config
- Delete or modify checkpoints
- Write to any file other than experiment-context.md and prompt files

**Always:**
- Base analysis on VP baseline from experiment-context.md
- Include actual numbers in diagnosis (not just "loss spiked")
- Record all observations in experiment-context.md
- Produce a prompt file that gives the next session complete context
- Stay responsive to user commands ("check now", "change frequency", "what's the status")

## Integration

- **spml:training-handoff** — Produces the context and prompt that starts this skill
- **spml:training-resume** — Consumes the recovery/completion prompts this skill produces
- **spml:diagnostics** — This skill's diagnosis mode uses similar evidence-gathering patterns, but from logs rather than live execution
