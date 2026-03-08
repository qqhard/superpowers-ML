# Watchdog Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three new skills (ml-training-handoff, ml-watchdog, ml-training-resume) and update existing skills/docs to support long-running ML task monitoring through a session chain architecture.

**Architecture:** Independent agent sessions connected through prompt files + a shared experiment-context.md file. Training scripts are production-deployable core code. Watchdog is read-only. All prompts are framework-agnostic.

**Design doc:** `docs/plans/2026-03-08-watchdog-agent-design.md`

---

### Task 1: Update main design doc — project structure

**Files:**
- Modify: `docs/plans/2026-03-06-superpowers-ml-design.md`

**Step 1: Add long-running workflow skills to project structure**

In the project structure tree (Section 2), after the `ml-verification/` line and before `# Pluggable framework knowledge`, add:

```markdown
    # Long-running task monitoring (session chain)
    ml-training-handoff/             # Main Agent → Watchdog: generates training script + context + prompts
    ml-watchdog/                     # Watchdog Agent: read-only monitoring of long-running tasks
    ml-training-resume/              # Recovery/Completion Agent: reads context, decides next workflow step
```

**Step 2: Commit**

```bash
git add docs/plans/2026-03-06-superpowers-ml-design.md
git commit -m "docs: add watchdog skills to project structure"
```

---

### Task 2: Update main design doc — workflow diagram

**Files:**
- Modify: `docs/plans/2026-03-06-superpowers-ml-design.md`

**Step 1: Update the full pipeline in Section 3**

Replace the existing full pipeline diagram with:

```
ml-brainstorming
    |
    Output: experiment design (includes validation scope + long-running confirmation)
    |
ml-experiment-planning
    |
    Output: shared scaffold definition + atomic subtask list
    |
ml-subagent-dev (execute subtasks one by one)
    |
    Each subtask:
    +-- 1. Function/operator-level unit test (deterministic code)
    +-- 2. Implementation code
    +-- 3. validation-pyramid (dynamically orchestrated per config)
    |       L0 -> L1 -> L2 -> L3
    |       Any layer fails -> ml-diagnostics
    |       Diagnostics support hierarchical decomposition (whole -> substructure -> operator)
    +-- 4. Spec review (does implementation match experiment design?)
    +-- 5. Record conclusion (metric data + effective/ineffective/needs further study)
    |
    Task needs long-running phase?
    |
    +-- No -> ml-verification (all subtasks complete)
    |
    +-- Yes -> ml-training-handoff
                |
                Output: training script + logs + experiment-context.md + watchdog-prompt.md
                |
                User chooses: separated execution or combined execution
                |
                [User starts training + Watchdog session]
                |
              ml-watchdog (independent session, read-only monitoring)
                |
                +-- Normal completion -> completion-prompt.md
                +-- Anomaly detected -> recovery-prompt.md
                |
              ml-training-resume (independent session)
                |
                +-- From completion -> ml-verification
                +-- From recovery -> back to appropriate stage
                    (code fix / replan / rebrainstorm)
                |
ml-verification (all subtasks complete)
    |
    Output: conclusion summary + recommendations
    |
Human decides: finish / add subtasks / new brainstorm round
```

**Step 2: Add Section 3.8 — ml-training-handoff**

After Section 3.7 (ml-verification), add:

```markdown
### 3.8 ml-training-handoff

**New skill.** Bridges VP validation (minute-level) and long-running execution (hours/days).

**Trigger:** After ml-subagent-dev completes subtasks and VP passes, when task includes long-running phase.

**Produces:**
1. Training script — core code, zero agent dependency, production-deployable. Includes dual output logging: tqdm progress bar (terminal, for humans) + JSONL metrics file (for Watchdog).
2. experiment-context.md — full experiment context: design, VP baseline, training config, code state.
3. watchdog-prompt.md — short, framework-agnostic prompt for starting a Watchdog session.

**User options:**
- Separated execution: user runs training independently, starts Watchdog in a separate session
- Combined execution: user starts Watchdog session which launches training and monitors

### 3.9 ml-watchdog

**New skill.** Watchdog Agent behavior definition — read-only monitoring of long-running tasks.

**Mechanism:** Periodically reads JSONL metrics log, compares against VP baseline from experiment-context.md, detects anomalies through trend analysis and pattern recognition (not just threshold alerts).

**Adaptive frequency:** High at start (catch startup issues), normal during steady state, higher near completion. User can override anytime.

**Output depends on execution mode:**
- Separated: silent when normal, speaks only on anomaly
- Combined: periodic progress reports + anomaly diagnostics

**On anomaly:** Writes diagnosis to experiment-context.md, produces recovery-prompt.md.
**On completion:** Writes summary to experiment-context.md, produces completion-prompt.md.

**Boundary:** Observes only. Never stops training, modifies code, or adjusts hyperparameters.

### 3.10 ml-training-resume

**New skill.** Recovery or Completion Agent entry point.

**From recovery:** Agent reads experiment-context.md (including Watchdog diagnosis), autonomously judges which workflow stage to return to: code fix, hyperparameter adjustment, replan, or rebrainstorm.

**From completion:** Agent analyzes final results vs hypothesis, enters ml-verification flow.

**Key:** Agent decides rollback level from evidence, not from Watchdog suggestion.
```

**Step 3: Commit**

```bash
git add docs/plans/2026-03-06-superpowers-ml-design.md
git commit -m "docs: add long-running workflow stages to core workflow"
```

---

### Task 3: Update main design doc — Phase 3 scope and new principle

**Files:**
- Modify: `docs/plans/2026-03-06-superpowers-ml-design.md`

**Step 1: Add new core principle**

In Section 1 Core principles, after principle 8, add:

```markdown
9. **Training scripts are core code.** Like model and training loop code, generated training scripts are independently deployable to production with zero agent dependency. The Watchdog monitoring protocol (prompt + experiment-context.md + JSONL logs) is framework-agnostic — any LLM agent can execute it.
```

**Step 2: Update Phase 3 scope**

In Section 6 Phase 3, update the bullet list to:

```markdown
- `vp-e2e-pipeline` skill
- `ml-diagnostics` skill (three core questions + hierarchical decomposition)
- `ml-subagent-dev` skill (ML-adapted review criteria)
- `ml-verification` skill (conclusion summary + recommendations)
- `ml-training-handoff` skill (generates training script + context + Watchdog prompt)
- `ml-watchdog` skill (read-only long-running task monitoring)
- `ml-training-resume` skill (recovery/completion entry point, rollback decision)
```

Update Phase 3 delivery criteria to:

```markdown
**Delivery criteria:** A complete ML experiment can go through the full pipeline: brainstorm -> plan -> execute -> validate -> long-running training with monitoring -> conclude.
```

**Step 3: Commit**

```bash
git add docs/plans/2026-03-06-superpowers-ml-design.md
git commit -m "docs: add watchdog principle and update Phase 3 scope"
```

---

### Task 4: Create ml-training-handoff skill

**Files:**
- Create: `skills/ml-training-handoff/SKILL.md`

**Step 1: Write the skill**

```markdown
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
```

**Step 2: Commit**

```bash
git add skills/ml-training-handoff/SKILL.md
git commit -m "feat: add ml-training-handoff skill"
```

---

### Task 5: Create ml-watchdog skill

**Files:**
- Create: `skills/ml-watchdog/SKILL.md`

**Step 1: Write the skill**

```markdown
---
name: ml-watchdog
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

- User has pasted a Watchdog prompt from ml-training-handoff
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

Your job: compare results against the hypothesis, run ml-verification, and present conclusions to the user.
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

- **spml:ml-training-handoff** — Produces the context and prompt that starts this skill
- **spml:ml-training-resume** — Consumes the recovery/completion prompts this skill produces
- **spml:ml-diagnostics** — This skill's diagnosis mode uses similar evidence-gathering patterns, but from logs rather than live execution
```

**Step 2: Commit**

```bash
git add skills/ml-watchdog/SKILL.md
git commit -m "feat: add ml-watchdog skill"
```

---

### Task 6: Create ml-training-resume skill

**Files:**
- Create: `skills/ml-training-resume/SKILL.md`

**Step 1: Write the skill**

```markdown
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
```

**Step 2: Commit**

```bash
git add skills/ml-training-resume/SKILL.md
git commit -m "feat: add ml-training-resume skill"
```

---

### ~~Task 7-10: REMOVED~~

> Tasks 7-10 originally modified existing skills (ml-subagent-dev, ml-verification, ml-brainstorming, ml-data-preparation) to "sense" the long-running phase. This was over-engineering — ML tasks are almost always long-running, so existing skills don't need explicit awareness. The handoff/watchdog/resume skills exist as independent workflow steps; routing is determined at the plan level, not embedded in each skill.

---

### Task 7: Final review

**Step 1: Verify all new skill files exist**

```bash
ls skills/ml-training-handoff/SKILL.md
ls skills/ml-watchdog/SKILL.md
ls skills/ml-training-resume/SKILL.md
```

Expected: all three files exist.

**Step 2: Verify cross-references are consistent**

Check that the three new skills reference each other correctly:
- ml-training-handoff references: ml-subagent-dev, ml-watchdog, ml-verification
- ml-watchdog references: ml-training-handoff, ml-training-resume, ml-diagnostics
- ml-training-resume references: ml-watchdog, ml-training-handoff, ml-verification, ml-brainstorming, ml-experiment-planning, ml-data-preparation, ml-diagnostics

**Step 3: Verify design doc references**

```bash
grep -c "ml-training-handoff\|ml-watchdog\|ml-training-resume" docs/plans/2026-03-06-superpowers-ml-design.md
```

Expected: multiple matches (project structure + workflow + Phase 3).

**Step 4: Verify existing skills are NOT modified**

```bash
git diff d41b29c -- skills/ml-subagent-dev/ skills/ml-verification/ skills/ml-brainstorming/ skills/ml-data-preparation/
```

Expected: no diff (these skills remain untouched).
