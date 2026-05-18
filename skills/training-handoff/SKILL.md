---
name: training-handoff
description: Use after VP passes when the task needs post-validation supervision — routes between watchdog (single run) and ml-iteration (N rounds), generating the appropriate protocol and startup prompt
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

## Preconditions

1. All enabled VP layers passed, baseline numbers recorded.
2. Design doc contains `review_criteria` block — at minimum the `metrics` sub-section. If missing, halt handoff and ask the user to add it to the design doc (this is the compass for both paths).

## Routing

After VP validation passes, ask the user to choose between two post-handoff paths:

> "This experiment is validated. Two ways to run from here:
>
> 1. **watchdog** — run training once, auto-restart on environment failures, async eval on new checkpoints. Choose this when the training script is ready to ship and you just want it supervised.
>
> 2. **ml-iteration** — run N short rounds (~10 min each). Each round a Researcher subagent modifies code, Supervisor trains + evaluates, Supervisor reviews against `review_criteria`, you can interject any time. Choose this when the training script still has rough edges (speed, log format, stability, partially satisfied metrics).
>
> Which?"

**Default suggestion:**
- `review_criteria` has at least one unmet dimension in VP results → suggest `ml-iteration`.
- All `review_criteria` already passed in VP L1 → suggest `watchdog`.
- Brainstorming explicitly flagged single-run intent → suggest `watchdog`.

The choice branches the remainder of the handoff into one of the two sub-flows below.

## Watchdog Branch

(invoked when user picks watchdog)

### Checklist

1. **Verify VP completion** — all enabled layers passed with actual numbers
2. **Verify directory layout** — experiment artifacts are co-located
3. **Verify training script readiness** — check existing script against production requirements
4. **Write experiment-context.md** — full context for downstream sessions
5. **Write watchdog-prompt.md** — prompt for Watchdog session
6. **Present launch instructions** — how to start the Watchdog session

### Step 1: Verify VP Completion

Confirm all VP layers that were enabled in the brainstorm design doc have passed. Record the actual metrics as VP baseline — the Watchdog will use these as reference.

### Step 2: Verify Directory Layout

Check that the experiment follows the co-located layout convention from `spml:experiment-planning`:

- Training script, tests, and outputs are under the same experiment directory
- Handoff artifacts (experiment-context.md, watchdog-prompt.md) will be written to that same directory

**Not a hard gate** — the user may have reasons for a different layout. If artifacts would be written outside the training script's directory, flag this to the user and ask whether to proceed or reorganize.

### Step 3: Verify Training Script Readiness

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

#### Upstream: Production Script Requirements

These requirements should be part of the experiment plan so ml-subagent-dev implements them from the start. If you find yourself frequently patching scripts at handoff, the fix is in `spml:experiment-planning`, not here. Key requirements to include in plans:

- Console output: tqdm (preferred) or controlled print, minute-level frequency, carrying key metrics (loss, lr, etc.)
- File output: detailed metrics written to file (loss, grad_norm, lr, mfu, memory, step_time), for Agent exploration
- Checkpoint: periodic save with configurable interval, resume support
- Step-based evaluation cadence
- `full validation` as the default evaluation scope unless explicitly overridden
- One shared evaluator core across checkpoint-based and in-memory evaluation
- Mode-aware evaluation errors and observability from the start

#### Evaluation Readiness Checks

When the experiment includes evaluation, verify these handoff-specific items:

- eval_command is defined and runnable (required for Watchdog async evaluation)
- checkpoint_dir is configured and accessible
- In-training evaluation fires at the planned step cadence (test by inspecting early log output or config)
- Checkpoint-based evaluation exists as an independent entrypoint

For the full evaluation requirements (observability, failure handling, mode-aware errors), see `experiment-planning` → Evaluation Planning Requirements. Gaps found here mean the upstream plan was incomplete — flag to user, do NOT silently rewrite VP-validated code.

### Step 4: Write experiment-context.md

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

### Step 5: Write watchdog-prompt.md

```markdown
I need you to act as a Watchdog Agent, monitoring and shepherding a long-running ML task.

## Setup
1. Read `[experiment-dir]/experiment-context.md` for full experiment context, VP baseline, and watchdog mode
2. Locate the training log at `[experiment-dir]/outputs/train.log`
3. Locate the training script: `[exact launch command]`

## Your Behavior
Use the spml:watchdog skill. It will guide you through:
- Launching the training script
- Monitoring the training log for anomalies
- Restarting from checkpoint on environment failures
- Running async evaluation when new checkpoints appear
- Reporting any non-environment anomaly to the user with a diagnosis (no auto-fix)
- Recording interventions in experiment-context.md
- Notifying you when training finishes or encounters issues
```

### Step 6: Present Launch Instructions

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

## Iteration Branch

(invoked when user picks ml-iteration)

1. **Verify training script readiness** — same "Required" checks as watchdog branch (standalone runnable, log to file, fixed seeds).
2. **Verify directory layout** — same as watchdog branch (co-located artifacts).
3. **Extract iteration parameters** — from the design doc and user:
   - `max_rounds` — default 10, ask to override.
   - `time_limit` — per-round training budget. Default: `time_limit` used in VP L1 (typically 5–10 min).
   - `focused_files` — "which files do you expect to change most this iteration?" (soft boundary hint, can be empty).
   - `locked_files` — auto-populated: `eval_command` script path + core data loader path.
   - `initial_hints` — optional first-round Researcher hint.
3.5. **Profile dry-run** — if the design doc has `profile_command` (set when `metric_category == performance` or when `review_criteria.performance` is populated):

   ```bash
   cd <experiment_dir>
   <profile_command>
   ```

   Require: exit 0 + non-empty stdout. If the design doc recorded `profile_command: TODO`, STOP and ask the user to provide one before re-running handoff. Mirror the message from `autoresearch-handoff` Step 4.5.

3.6. **Kernel parity dry-run** — if the design doc has `kernel_targets` non-empty:

   For each target: check the `new_module` file exists. If absent, STOP and tell the user to create it as a re-export of the baseline module so baseline parity is trivially passable, then re-run handoff. When all targets resolve, for each target execute the inline Python heredoc from `ml-iteration/SKILL.md` Step 2 (the parity check block), substituting the target's values and `$experiment_dir` for the placeholders.

   Trivial-pass expected. Failure → STOP with the PARITY_FAIL line. Same semantics as `autoresearch-handoff` Step 4.6.
4. **Generate `iteration-protocol.md`** at the experiment directory — template below.
5. **Generate `experiences.md`** with initial header (mode: iteration, status: not_started, rounds: 0).
6. **Generate `iteration-prompt.md`** — a startup prompt the user pastes into a new session to launch `ml-iteration`.
7. **Present launch instructions** — "Start a new Claude session in this directory, paste iteration-prompt.md."

### iteration-protocol.md Template

````markdown
---
mode: iteration
experiment: {experiment_name}
max_rounds: {max_rounds}
time_limit: {time_limit}
train_command: {train_command from VP L1}
eval_command: {eval_command from VP L1}
---

## Review criteria
{copied verbatim from design doc}

## Modification boundary (soft)
- focused_files: {focused_files}
- locked_files: {locked_files}
- Other: soft — Researcher may modify; will be recorded in experiences.md

## Initial hints
{initial_hints or "(none)"}

<!-- Include only if design doc has profile_command -->
## Profile
- command: {profile_command from design doc}
<!-- end Profile -->

<!-- Include only if design doc has kernel_targets non-empty. The yaml fence below is intentional — Supervisor parses kernel_targets from the first ```yaml fence in the file. -->
## Kernel Targets

```yaml
kernel_targets:
  - name: {readable name}
    new_module: {module:attr}
    baseline_module: {module:attr}
    fixture: {module:attr}
    tolerance: { atol: {float}, rtol: {float} }
  # repeat per target
```
<!-- end Kernel Targets -->
````

### iteration-prompt.md Template

```markdown
Run ml-iteration at {experiment_dir}.

Protocol: {experiment_dir}/iteration-protocol.md
Experiences log: {experiment_dir}/experiences.md

Use the `spml:ml-iteration` skill.
```

## Integration

- **spml:ml-subagent-dev** — Triggers handoff after VP passes (when long-running phase needed)
- **spml:watchdog** — The Watchdog prompt references this skill's behavior
- **spml:verification** — Skipped at handoff; entered later via re-entry on experiment directory
