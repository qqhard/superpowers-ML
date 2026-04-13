---
name: watchdog
description: Use when monitoring a single long-running ML training run — restarts from checkpoint on environment failures, runs async evaluation on new checkpoints, and surfaces anomalies for the user to handle
---

# ML Watchdog

## Overview

Active shepherd for a single long-running ML training run. Periodically reads training logs, detects anomalies through trend analysis and pattern recognition, and restarts from the latest checkpoint when the environment fails.

**Core principle:** Keep one training run healthy. Watchdog's only intervention is restart-from-checkpoint on environment failures. Everything else (parameter tuning, code fixes, multi-round iteration) is out of scope — surface it to the user or hand off to `ml-iteration`.

## Shared Patterns

This skill uses the following primitive patterns — see `skills/_ml-loop-primitives/` for details:
- `scheduling-safety-net.md` — monitoring loop timer discipline.

<HARD-GATE>
## Monitoring Loop Mechanism

You MUST use the Bash tool to execute `sleep <seconds>` as the ONLY way to wait between checks.

After each sleep returns, immediately proceed to the next check — do NOT wait for user input.

**Prohibited:**
- Outputting "I'll check in N minutes" and then stopping
- Writing a standalone monitoring/watchdog script that runs its own loop or scheduling (Python, shell, or any other). Inline one-liners for log parsing are fine.
- Using /loop or any external scheduling mechanism
- Asking the user to remind you to check

**Required execution pattern:**
1. Bash tool: `sleep <interval>`  (see Monitoring Loop step 1 for interval values)
2. Bash tool: `tail -20 <log_file>`  (read latest log lines)
3. Analyze output, report status
4. If anomaly → diagnose and act (restart on environment failure, otherwise surface to user)
5. Go to step 1
</HARD-GATE>

## Scope

Watchdog keeps a single training run healthy. Its only intervention is restarting from the latest checkpoint after an environment failure. It does not change parameters, fix code, or decide what to do next — those belong in `ml-iteration` or `autoresearch`.

If you need iterative parameter or code changes, stop watchdog and re-run `training-handoff` to pick `ml-iteration` instead.

## When to Use

- User has pasted a Watchdog prompt from `training-handoff` (watchdog branch).
- A single long-running training run needs supervision, and the user does not want N-round iteration.

## When Not to Use

- User wants parameter tuning, code fixes, or multiple training rounds → use `ml-iteration`.
- User wants metric-driven search with Fixed/Variable file partitions → use `autoresearch`.

## Startup

1. **Read experiment-context.md** — understand experiment design, VP baseline, expected behavior, training config
2. **Verify log file exists** — check that the training log file is at the expected path
3. **Locate training script** — confirm the training script path and launch command from experiment-context.md
4. **Launch training** — run the training script via Bash tool, then enter monitoring loop

## Problem Classification

Two outcomes only:

- **Environment problem** (OOM killer, NCCL timeout, hardware error, disk full, SIGKILL, hang past baseline × 10) → restart from latest checkpoint. No retry limit; if crashes persist (e.g., 5+ within 30 minutes), surface a warning but keep retrying.
- **Anything else** (code bug, wrong metric trend, NaN in inputs, plateau past VP baseline) → report to the user, do not auto-fix. Write a diagnosis to `experiment-context.md` and notify the user.

### Tier 1: Environment Problems

Process died from external causes, not code bugs.

Examples:
- OOM killer (exit code 137)
- NCCL timeout / network errors
- Hardware GPU errors (not caused by code)
- Disk full
- SIGKILL / SIGTERM from external source
- Deadlock / hang (process alive but no output for significantly longer than baseline step interval)

**Action:** Restart training script from latest checkpoint. No code changes. No retry limit — keep restarting. After repeated crashes (e.g., 5+ within 30 minutes), notify the user that environment instability is persisting but continue retrying.

## Monitoring Loop

```
loop {
    1. Bash tool: `sleep <interval_seconds>`
       - Normal: 120-300s (2-5 min)
       - Post-restart / post-anomaly: 60s for 5 cycles, then back to normal
    2. Bash tool: `tail -20 <log_file>` (each new log line = heartbeat; format is human-readable text, not JSONL)
    3. Check for new lines since last check:
       a. New lines → parse metrics, go to step 4
       b. No new lines → Bash tool: `ps aux | grep <training_script>`
          - Process dead → read exit code → classify (environment → restart; otherwise → surface to user)
          - Process alive → compare silence duration vs step baseline
            - Startup grace period (first 15 min or until 3 logged steps): do not classify silence as hang
            - Within baseline → continue (go to step 1)
            - Exceeds 10x baseline → kill process → classify (environment hang → restart; suspected code issue → surface to user)
    3.5. Async evaluation check (skip entirely if no eval_command in experiment-context.md):
         a. Check if background eval subagent has returned:
            - Returned → read summary, append to experiment-context.md Evaluation History
            - Set eval_subagent_running=false, update last_evaluated_checkpoint
         b. Scan checkpoint directory for newest checkpoint (Bash: `ls -t <checkpoint_dir> | head -1`)
         c. Compare with last_evaluated_checkpoint:
            - Same → skip
            - Newer → check all preconditions:
              (1) eval_subagent_running = false
              (2) eval_paused = false
              (3) Free GPU available (Bash: `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` — at least one GPU below 90%)
            - All met → spawn eval subagent (see Async Evaluation section), set eval_subagent_running=true
            - Any failed → log reason ("eval busy" / "eval paused" / "no free GPU"), continue
    4. Analyze metrics:
       a. Sanity: NaN, Inf, negative loss, zero gradient
       b. Baseline comparison vs VP ranges in experiment-context.md
       c. Trend: loss decreasing? grad_norm stable?
       d. Anomaly patterns: spike, plateau, divergence, sudden shift
    5. Classify:
       - NORMAL → update step time baseline, output one-line progress, go to step 1
       - ANOMALY → diagnose, act (restart on environment failure, otherwise surface to user), record in experiment-context.md
       - COMPLETE → enter completion mode
    6. After any restart → enter intensive observation (60s interval, 5 cycles)
}
```

## Restart Mechanism

Obtain the training script path and launch command from experiment-context.md (written by training-handoff). Restart = re-run the same command via Bash tool. The training script's built-in checkpoint resume handles continuation from the latest saved state.

After restart, enter **intensive observation** (1-minute interval, 5 cycles) to confirm training resumes normally.

## Async Evaluation

Watchdog spawns background evaluation subagents when new checkpoints are detected. This keeps training uninterrupted and the Watchdog's context clean.

**Preconditions (all must be true to spawn):**
1. `eval_command` exists in experiment-context.md (otherwise evaluation is disabled)
2. New checkpoint detected (newer than last_evaluated_checkpoint)
3. eval_subagent_running = false (previous eval finished)
4. eval_paused = false (user hasn't paused)
5. Free GPU available (at least one GPU below 90% utilization via nvidia-smi)

**Eval subagent prompt template:**

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

**Dispatch:** Use Agent tool with `run_in_background: true`.

**On return:** Watchdog reads the summary, appends to experiment-context.md Evaluation History, sets eval_subagent_running=false, updates last_evaluated_checkpoint.

**State variables (transient, not persisted):**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| last_evaluated_checkpoint | string | null | Path/identifier of last evaluated checkpoint |
| eval_subagent_running | bool | false | Whether an eval subagent is currently running |
| eval_paused | bool | false | User has paused evaluation |

## Progress Reports

During normal monitoring, output periodic progress:
```
[14:10] step 2000/10000 (20%) — loss=0.52, MFU=45%, grad_norm=1.8 [estimated 3:20 remaining]
```

## Diagnosis Mode

Triggered by ANOMALY classification.

### Step 1: Collect Evidence

From the training log, gather:
- Metrics at the anomaly point and preceding lines
- Which metrics changed simultaneously
- Whether the change was sudden or gradual
- Step number and timestamp of onset

### Step 2: Contextualize

Compare against experiment-context.md:
- Is this metric outside the VP baseline range?
- Does the training config explain this? (e.g., LR schedule transition)
- Has a similar issue been recorded in Diagnosis History? (recurring problem)

### Step 3: Classify and Act

Based on evidence, classify the problem as environment failure or other. Environment failures → restart from latest checkpoint. Everything else → surface to the user with a written diagnosis (see Problem Classification section above).

### Step 4: Record

Append to experiment-context.md Diagnosis History:

```markdown
### Issue #N — step [step_number]
- Symptom: [what happened, with numbers]
- Classification: [environment / other]
- Context: [what was expected at this training phase]
- Co-occurring signals: [other metrics that moved]
- Action taken: [restart / reported to user]
- Checkpoint reference: [nearest checkpoint before/after anomaly]
```

## Completion Mode

Triggered when training reaches final step or early stop condition.

### Step 1: Final Summary

Read the training log and summarize:
- Final metric values
- Training trajectory (was convergence smooth?)
- Total training time
- Any anomalies that occurred and resolved during training
- All interventions taken (restarts)
- Evaluation results summary (from Evaluation History in experiment-context.md)

### Step 2: Compare Against Expectations

- Final loss vs what was expected from VP baseline
- MFU consistency vs VP baseline
- Any metric drift from VP baseline

### Step 3: Update experiment-context.md

```markdown
## Watchdog Status
- Status: completed
- Total steps: [N]
- Total time: [duration]
- Final metrics: loss=[val], ...
- Anomalies during training: [count, brief summary]
- Interventions: [count of restarts]
```

### Step 4: Notify User

```
Training complete. [total_steps] steps in [duration].
Final loss: [val].
Interventions: [N restarts].

Start a new session on the experiment directory to analyze results and conclude the experiment.
```

## Common Anomaly Patterns

| Pattern | Indicators | Response |
|---------|-----------|---------|
| Loss spike | Sudden loss increase > 3x recent average | Surface to user (process crash → restart) |
| Loss plateau | Loss unchanged for > 5% of total steps | Surface to user |
| Gradient explosion | grad_norm > 10x VP baseline | Surface to user |
| Gradient vanishing | grad_norm < 0.01x VP baseline | Surface to user |
| NaN/Inf | Any NaN or Inf in metrics | Surface to user |
| MFU drop | MFU drops > 20% from VP baseline | Environment (thermal/I/O) → surface; if process crashes, restart |
| Stale log | No new entries for > 10x baseline step time | Environment hang → restart |
| CUDA OOM | RuntimeError: CUDA out of memory | Surface to user (parameter change needed) |
| Process exit 137 | Killed by OOM killer | Environment → restart |
| NCCL error | NCCL timeout or connection reset | Environment → restart |

## Red Flags

**Never:**
- Modify training code, configuration, or parameters
- Delete or modify checkpoints
- Spawn sub-agents to fix issues
- Ignore Diagnosis History from previous interventions

**Always:**
- Base analysis on VP baseline from experiment-context.md
- Include actual numbers in diagnosis (not just "loss spiked")
- Record ALL interventions (restarts) in experiment-context.md
- When uncertain whether a problem is environmental, surface to the user rather than restarting blindly
- Stay responsive to user commands ("check now", "change frequency", "what's the status")
- Respond to evaluation commands ("pause eval", "resume eval", "eval status") — pause/resume sets eval_paused flag; status reports last evaluated checkpoint, whether eval is running, and skipped checkpoint count

## Integration

- **spml:training-handoff** — Produces the context and prompt that starts this skill
- **spml:ml-iteration** — Use this instead when parameter tuning, code fixes, or multi-round iteration is needed
- **spml:diagnostics** — Users can invoke this directly for deeper analysis when watchdog surfaces an anomaly
