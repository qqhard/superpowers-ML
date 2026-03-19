# Watchdog Active Shepherd Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Watchdog skill from a read-only observer to an active shepherd that can restart training, fix simple parameter issues, and spawn sub-agents for complex problems.

**Architecture:** Rewrite the Watchdog skill with three operating modes (Monitor/Guardian/Autonomous) and three-tier problem classification. Update training-handoff to use human-readable logs instead of JSONL and include mode configuration. Update training-resume for sub-agent spawn compatibility.

**Tech Stack:** Markdown skill definitions (no code changes)

**Spec:** `docs/superpowers/specs/2026-03-17-watchdog-active-shepherd-design.md`

---

### Task 1: Rewrite Watchdog skill

**Files:**
- Rewrite: `skills/watchdog/SKILL.md`

The entire file needs rewriting. The new version replaces read-only observation with active intervention capabilities.

- [ ] **Step 1: Read current file**

Read `skills/watchdog/SKILL.md` to confirm current content before rewriting.

- [ ] **Step 2: Write the new Watchdog skill**

Replace the entire content of `skills/watchdog/SKILL.md` with the following:

```markdown
---
name: watchdog
description: Use when monitoring a long-running ML task — active shepherd with three operating modes, automatic restart, parameter fixing, and sub-agent spawning for complex issues
---

# ML Watchdog

## Overview

Active shepherd for long-running ML tasks. Periodically reads training logs, detects anomalies through trend analysis and pattern recognition, and takes action based on the configured operating mode — from reporting only (Monitor) to fully autonomous recovery (Autonomous).

**Core principle:** Keep training running. The Watchdog classifies problems into three tiers and responds with the minimum intervention needed — restart for environment crashes, parameter adjustment for simple issues, and sub-agent delegation for complex problems.

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
5. **Launch training** — run the training script via Bash tool, then enter monitoring loop

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

**Action (Guardian/Autonomous):** Restart training script from latest checkpoint. No code changes. No retry limit — keep restarting. After repeated crashes (e.g., 5+ in a short period), notify the user that environment instability is persisting but continue retrying.

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
    1. Sleep (interval)
    2. Read latest lines from training log (tail, not full re-read)
    3. Check for new lines:
       a. New lines present → parse metrics, analyze
       b. No new lines → check process alive
          - Process dead → read exit code → classify problem → execute action
          - Process alive → assess hang (compare silence duration vs step baseline)
            - Within baseline → continue waiting
            - Exceeds baseline significantly (e.g., 10x) → kill process → classify problem
    4. If new lines, analyze:
       a. Sanity: NaN, Inf, negative loss, zero gradient
       b. Baseline comparison: current metrics vs VP baseline ranges
       c. Trend: is loss decreasing? gradient norm stable? MFU consistent?
       d. Anomaly patterns: spike, plateau, divergence, sudden shift
    5. Classify state:
       - NORMAL: metrics within expected ranges, healthy trends → update step time baseline, report progress, continue
       - ANOMALY: deviation detected → classify problem tier → execute tier action
       - COMPLETE: training finished (final step reached or early stop triggered) → enter completion mode
    6. All interventions → record in experiment-context.md
    7. After restart → enter intensive observation period
}
```

## Polling and Hang Detection

**Log as heartbeat:** Training scripts output one line per step/epoch with key metrics (loss, lr, grad norm, etc.). Each new log line is a heartbeat signal. The format is human-readable text — not JSONL.

**Polling interval:**
- Normal: 2–5 minutes (sampling, not every step)
- Post-anomaly / post-restart: 1 minute for 5 cycles, then back to normal
- Must use Bash tool `sleep` to implement intervals — ensures the loop runs continuously and does not stall

**Hang detection:** During normal monitoring, observe step intervals and build a baseline of typical step duration. If no new log line appears for significantly longer than the baseline (e.g., 10x typical step duration) and the process is still alive, judge the process as hung. Kill the process and classify the problem:
- Likely environment (deadlock, NCCL hang) → Tier 1, restart
- Possibly code issue → Escalate to Tier 2 or Tier 3

**Step baseline:** Calculated from observed intervals between log lines during normal operation. Updated continuously. At startup (before baseline is established), use a generous timeout (e.g., 15 minutes) before judging a hang.

## Restart Mechanism

Obtain the training script path and launch command from experiment-context.md (written by training-handoff). Restart = re-run the same command via Bash tool. The training script's built-in checkpoint resume handles continuation from the latest saved state.

After restart, enter **intensive observation** (1-minute interval, 5 cycles) to confirm training resumes normally.

## Progress Reports

During normal monitoring, output periodic progress:
```
[14:10] step 2000/10000 (20%) — loss=0.52, grad_norm=1.8 [estimated 3:20 remaining]
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

Read `[path]/experiment-context.md` for the full context including:
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
```

- [ ] **Step 3: Verify the rewrite**

Read back `skills/watchdog/SKILL.md` and confirm:
- Frontmatter `name` is `watchdog`, description updated
- No HARD-GATE about read-only behavior
- Three operating modes documented (Monitor/Guardian/Autonomous)
- Three-tier problem classification with per-mode actions
- Monitoring loop includes process liveness check and hang detection
- Restart mechanism references experiment-context.md for script path
- Tier 2 parameter modification flow documented
- Tier 3 sub-agent spawn via Agent tool documented
- Log format is human-readable (no JSONL references)
- No separated/combined execution mode (always combined)

- [ ] **Step 4: Commit**

```bash
git add skills/watchdog/SKILL.md && git commit -m "feat: rewrite watchdog skill — active shepherd with three operating modes and three-tier problem classification"
```

---

### Task 2: Rewrite training-handoff skill

**Files:**
- Rewrite: `skills/training-handoff/SKILL.md`

Major changes: JSONL → human-readable logs, add watchdog_mode to experiment-context.md template, rewrite watchdog-prompt.md template, remove separated execution mode.

- [ ] **Step 1: Read current file**

Read `skills/training-handoff/SKILL.md` to confirm current content.

- [ ] **Step 2: Write the new training-handoff skill**

Replace the entire content of `skills/training-handoff/SKILL.md` with the following:

```markdown
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

- **spml:ml-subagent-dev** — Triggers handoff after VP passes (when long-running phase needed)
- **spml:watchdog** — The Watchdog prompt references this skill's behavior
- **spml:verification** — Skipped at handoff; entered later via resume
```

- [ ] **Step 3: Verify the rewrite**

Read back `skills/training-handoff/SKILL.md` and confirm:
- Frontmatter description updated (mentions human-readable logging)
- HARD-GATE references log output (not JSONL)
- No JSONL references anywhere in the file
- Log file code example uses human-readable format (not `json.dumps`)
- experiment-context.md template includes `watchdog_mode` field and `Launch command` field
- watchdog-prompt.md template references spml:watchdog skill, no "DO NOT modify code" constraint
- No separated/combined execution mode — single launch path
- Smoke test verifies human-readable log (not JSONL parsing)

- [ ] **Step 4: Commit**

```bash
git add skills/training-handoff/SKILL.md && git commit -m "feat: rewrite training-handoff — human-readable logs, watchdog mode config, active shepherd prompt"
```

---

### Task 3: Update training-resume skill

**Files:**
- Modify: `skills/training-resume/SKILL.md`

Changes: Update `metrics.jsonl` references to human-readable log file. Add behavior for when spawned by Watchdog sub-agent (no VP, fix and restart).

- [ ] **Step 1: Read current file**

Read `skills/training-resume/SKILL.md` to confirm current content.

- [ ] **Step 2: Update JSONL references**

Replace all references to `metrics.jsonl` with the human-readable log file:

1. Line 12 (`experiment-context.md and metrics.jsonl`):
   - Change to: `experiment-context.md and the training log`

2. Line 39 (`read metrics.jsonl directly`):
   - Change to: `read the training log directly`

3. Line 108 (`Read metrics.jsonl for:`):
   - Change to: `Read the training log for:`

- [ ] **Step 3: Add Watchdog sub-agent spawn behavior**

After the "## When to Use" section (line 17), add a new section:

```markdown
## When Spawned by Watchdog

When this skill is invoked by a Watchdog sub-agent (Autonomous mode, Tier 3 problem):
- The Watchdog has already written a diagnosis to experiment-context.md
- Follow the normal Recovery Path below
- Do NOT run VP — trust that initial VP validation covers code correctness
- After fixing the issue, restart the training script (launch command is in experiment-context.md)
- The Watchdog will resume monitoring after this agent completes
```

- [ ] **Step 4: Verify changes**

Read back `skills/training-resume/SKILL.md` and confirm:
- No `metrics.jsonl` references remain
- New "When Spawned by Watchdog" section is present after "When to Use"
- Rest of the file is unchanged

- [ ] **Step 5: Commit**

```bash
git add skills/training-resume/SKILL.md && git commit -m "feat: update training-resume — human-readable log refs, Watchdog sub-agent spawn behavior"
```

---

### Task 4: Update original design doc references

**Files:**
- Modify: `docs/plans/2026-03-08-watchdog-agent-design.md`

Add a note at the top pointing to the new spec, so readers know the design has evolved.

- [ ] **Step 1: Read the top of the design doc**

Read the first 10 lines of `docs/plans/2026-03-08-watchdog-agent-design.md`.

- [ ] **Step 2: Add superseded note**

After the title line, add:

```markdown
> **Note:** This document describes the original read-only Watchdog design. The Watchdog has since been redesigned as an active shepherd — see `docs/superpowers/specs/2026-03-17-watchdog-active-shepherd-design.md` for the current design.
```

- [ ] **Step 3: Commit**

```bash
git add docs/plans/2026-03-08-watchdog-agent-design.md && git commit -m "docs: add superseded note to original watchdog design doc"
```

---

### Task 5: Final review

**Files:**
- Read: all modified files

Cross-reference check to verify consistency.

- [ ] **Step 1: Verify cross-references**

Check that:
- `skills/watchdog/SKILL.md` Integration section references `spml:training-handoff` and `spml:training-resume`
- `skills/training-handoff/SKILL.md` Integration section references `spml:watchdog`
- `skills/training-resume/SKILL.md` Integration section references `spml:watchdog`
- experiment-context.md template in training-handoff includes `watchdog_mode` and `Launch command`
- watchdog-prompt.md template in training-handoff references `spml:watchdog` skill

- [ ] **Step 2: Search for stale JSONL references**

```bash
grep -r "JSONL\|jsonl\|metrics\.jsonl" skills/watchdog/ skills/training-handoff/ skills/training-resume/
```

Expected: No matches.

- [ ] **Step 3: Search for stale read-only references**

```bash
grep -r "read-only\|read only\|Watch, don't act\|DO NOT.*modify\|DO NOT.*stop\|DO NOT.*adjust" skills/watchdog/ skills/training-handoff/
```

Expected: No matches referencing the old read-only constraint. (Watchdog Red Flags section may have "Never: Modify training logic" which is correct — this refers to logic, not parameters.)

- [ ] **Step 4: Search for stale separated/combined mode references**

```bash
grep -r "separated\|combined mode\|separated mode" skills/watchdog/ skills/training-handoff/ skills/training-resume/
```

Expected: No matches.

- [ ] **Step 5: Commit any fixes found during review**

If any stale references were found, fix them and commit:

```bash
git add -A && git commit -m "fix: remove stale references found during final review"
```

If no fixes needed, skip this step.
