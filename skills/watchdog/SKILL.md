---
name: watchdog
description: Use when monitoring a long-running ML task — active shepherd with three operating modes, automatic restart, parameter fixing, and sub-agent spawning for complex issues
---

# ML Watchdog

## Overview

Active shepherd for long-running ML tasks. Periodically reads training logs, detects anomalies through trend analysis and pattern recognition, and takes action based on the configured operating mode — from reporting only (Monitor) to fully autonomous recovery (Autonomous).

**Core principle:** Keep training running. The Watchdog classifies problems into three tiers and responds with the minimum intervention needed — restart for environment crashes, parameter adjustment for simple issues, and sub-agent delegation for complex problems.

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
4. If anomaly → diagnose and act per operating mode
5. Go to step 1
</HARD-GATE>

## When to Use

- User has pasted a Watchdog prompt from training-handoff
- A long-running ML task needs monitoring (training, data processing, evaluation)

## Operating Modes

Three modes control Watchdog's intervention authority. Mode is read from `experiment-context.md` (`watchdog_mode` field), overridable at startup.

| Mode | Environment crash | Simple parameter issue | Complex issue |
|------|------------------|----------------------|---------------|
| **Monitor** | Report only | Report only | Report only |
| **Guardian** (default) | Auto-restart | Auto-fix + restart | Report to user |
| **Autonomous** | Auto-restart | Auto-fix + restart | Spawn sub-agent to fix + restart |

At startup, read the mode from experiment-context.md. Ask the user if they want to override it. If the user doesn't respond or just pastes the watchdog-prompt.md, use the preset value.

## Startup

1. **Read experiment-context.md** — understand experiment design, VP baseline, expected behavior, training config, watchdog mode
2. **Verify log file exists** — check that the training log file is at the expected path
3. **Confirm mode** — read `watchdog_mode` from experiment-context.md, offer user override
4. **Locate training script** — confirm the training script path and launch command from experiment-context.md
5. **Launch training** — run the training script via Bash tool (all modes, including Monitor), then enter monitoring loop

## Problem Classification

Watchdog is an LLM — use judgment, not rigid rules. The following are guiding examples, not exhaustive lists.

**Classification principle:** When uncertain, escalate upward (treat simple as complex). Never downgrade ambiguous problems.

### Tier 1: Environment Problems

Process died from external causes, not code bugs.

Examples:
- OOM killer (exit code 137)
- NCCL timeout / network errors
- Hardware GPU errors (not caused by code)
- Disk full
- SIGKILL / SIGTERM from external source
- Deadlock / hang (process alive but no output for significantly longer than baseline step interval)

**Action (Guardian/Autonomous):** Restart training script from latest checkpoint. No code changes. No retry limit — keep restarting. After repeated crashes (e.g., 5+ within 30 minutes), notify the user that environment instability is persisting but continue retrying.

**Action (Monitor):** Write diagnosis to experiment-context.md, generate recovery-prompt.md, notify user.

### Tier 2: Simple Parameter Problems

Needs numeric/config adjustment but no logic change.

Examples:
- CUDA OOM → reduce batch size or increase gradient accumulation steps
- Loss explosion (gradient explosion) → lower learning rate or add gradient clipping
- Extended plateau → adjust lr schedule
- Any numeric parameter that is clearly misconfigured

**Action (Guardian/Autonomous):**
1. Record current parameter value in experiment-context.md (before)
2. Modify the parameter in the training script or config file
3. Record new value and rationale in experiment-context.md (after)
4. Restart from checkpoint

**Action (Monitor):** Write diagnosis to experiment-context.md, generate recovery-prompt.md, notify user.

### Tier 3: Complex Problems

Requires logic changes or root cause is unclear.

Examples:
- Everything not in Tier 1 or 2
- Data issues (NaN in inputs, data loading errors)
- Model architecture problems (attention collapse, expert collapse)
- Code bugs

**Action (Autonomous):**
1. Write diagnosis to experiment-context.md
2. Generate recovery-prompt.md
3. Spawn sub-agent using Claude Code's Agent tool with instructions: read recovery-prompt.md, follow training-resume flow, fix issue, restart training
4. Sub-agent does NOT run VP — trusts that initial VP validation covers code correctness
5. Wait for Agent tool call to return, then resume monitoring loop

**Action (Guardian):** Write diagnosis to experiment-context.md, generate recovery-prompt.md, notify user. Wait for user to handle the issue.

**Action (Monitor):** Same as Guardian.

Note: Autonomous mode's sub-agent spawn introduces a Claude Code dependency. Monitor and Guardian modes remain framework-agnostic (only require bash access).

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
          - Process dead → read exit code → classify problem tier → act
          - Process alive → compare silence duration vs step baseline
            - Startup grace period (first 15 min or until 3 logged steps): do not classify silence as hang
            - Within baseline → continue (go to step 1)
            - Exceeds 10x baseline → kill process → classify problem tier (environment hang → Tier 1; possible code issue → Tier 2/3)
    4. Analyze metrics:
       a. Sanity: NaN, Inf, negative loss, zero gradient
       b. Baseline comparison vs VP ranges in experiment-context.md
       c. Trend: loss decreasing? grad_norm stable?
       d. Anomaly patterns: spike, plateau, divergence, sudden shift
    5. Classify:
       - NORMAL → update step time baseline, output one-line progress, go to step 1
       - ANOMALY → diagnose, classify tier, act per operating mode, record in experiment-context.md
       - COMPLETE → enter completion mode
    6. After any restart → enter intensive observation (60s interval, 5 cycles)
}
```

## Restart Mechanism

Obtain the training script path and launch command from experiment-context.md (written by training-handoff). Restart = re-run the same command via Bash tool. The training script's built-in checkpoint resume handles continuation from the latest saved state.

After restart, enter **intensive observation** (1-minute interval, 5 cycles) to confirm training resumes normally.

## Progress Reports

During normal monitoring, output periodic progress:
```
[14:10] step 2000/10000 (20%) — loss=0.52, MFU=45%, grad_norm=1.8 [estimated 3:20 remaining]
```

## Diagnosis Mode

Triggered by ANOMALY classification (any tier).

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

Based on evidence, classify the problem as Tier 1, 2, or 3, then execute the corresponding action for the current operating mode (see Problem Classification section above).

### Step 4: Record

Append to experiment-context.md Diagnosis History:

```markdown
### Issue #N — step [step_number]
- Symptom: [what happened, with numbers]
- Classification: Tier [1/2/3] — [environment/simple parameter/complex]
- Context: [what was expected at this training phase]
- Co-occurring signals: [other metrics that moved]
- Action taken: [restart / parameter change / sub-agent spawned / reported to user]
- Changes made: [if any, with before/after values]
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
- All interventions taken (restarts, parameter changes, sub-agent fixes)

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
- Interventions: [count and types — restarts, parameter changes, sub-agent fixes]
```

### Step 4: Produce completion-prompt.md

```markdown
A long-running ML task has completed successfully. Please analyze results and conclude the experiment.

Read `[experiment-dir]/experiment-context.md` for the full context including:
- Original experiment design and hypothesis
- VP baseline metrics
- Watchdog monitoring summary and interventions
- Final training metrics

Your job: compare results against the hypothesis, run verification, and present conclusions to the user.
```

### Step 5: Notify User

```
Training complete. [total_steps] steps in [duration].
Final loss: [val].
Interventions: [N restarts, N parameter changes, N sub-agent fixes].
To analyze results: open a new agent session and paste the contents of completion-prompt.md.
```

## Common Anomaly Patterns

| Pattern | Indicators | Typical Tier |
|---------|-----------|-------------|
| Loss spike | Sudden loss increase > 3x recent average | Tier 2 (lower LR) or Tier 1 (if process crashes) |
| Loss plateau | Loss unchanged for > 5% of total steps | Tier 2 (adjust LR schedule) |
| Gradient explosion | grad_norm > 10x VP baseline | Tier 2 (lower LR or add clipping) |
| Gradient vanishing | grad_norm < 0.01x VP baseline | Tier 3 (architecture issue) |
| NaN/Inf | Any NaN or Inf in metrics | Tier 3 (numerical instability) |
| MFU drop | MFU drops > 20% from VP baseline | Tier 1 (thermal throttle, I/O) |
| Stale log | No new entries for > 10x baseline step time | Tier 1 (hang/deadlock) |
| CUDA OOM | RuntimeError: CUDA out of memory | Tier 2 (reduce batch size) |
| Process exit 137 | Killed by OOM killer | Tier 1 (restart) |
| NCCL error | NCCL timeout or connection reset | Tier 1 (restart) |

## Red Flags

**Never:**
- Modify training logic (only numeric parameters in Tier 2)
- Delete or modify checkpoints
- Ignore Diagnosis History from previous interventions

**Always:**
- Base analysis on VP baseline from experiment-context.md
- Include actual numbers in diagnosis (not just "loss spiked")
- Record ALL interventions in experiment-context.md with before/after values
- Escalate upward when uncertain about problem classification
- Stay responsive to user commands ("check now", "change frequency", "what's the status", "switch to autonomous mode")

## Integration

- **spml:training-handoff** — Produces the context and prompt that starts this skill
- **spml:training-resume** — Invoked by sub-agents in Autonomous mode; consumes recovery/completion prompts in Monitor/Guardian mode
- **spml:diagnostics** — Sub-agents in Autonomous mode may invoke this for deeper analysis
