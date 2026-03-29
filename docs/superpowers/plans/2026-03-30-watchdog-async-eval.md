# Watchdog Async Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add async checkpoint evaluation to Watchdog via background subagents, with GPU availability checks and user pause/resume control.

**Architecture:** Two skill files modified — Watchdog gets a new monitoring loop step (3.5) plus supporting sections; training-handoff gets experiment-context.md template additions and a new checklist item.

**Tech Stack:** Markdown skill files

**Spec:** `docs/superpowers/specs/2026-03-30-watchdog-async-eval-design.md`

---

### Task 1: Add async evaluation to Watchdog monitoring loop

**Files:**
- Modify: `skills/watchdog/SKILL.md:124-149` (Monitoring Loop)
- Modify: `skills/watchdog/SKILL.md:260-270` (Red Flags — Always list)

- [ ] **Step 1: Insert step 3.5 in the monitoring loop**

In the monitoring loop code block (lines 124-149), insert the following after step 3 (the `Check for new lines since last check` block, ending at step 3b) and before step 4 (`Analyze metrics`):

```
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
```

- [ ] **Step 2: Add Async Evaluation section**

Add a new section after the "Restart Mechanism" section (after line 155) and before "Progress Reports" (line 158):

```markdown
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
```

- [ ] **Step 3: Add evaluation user commands**

In the "Red Flags" section, in the "**Always:**" list (around line 270), add the following bullet after the existing last item:

```markdown
- Respond to evaluation commands ("pause eval", "resume eval", "eval status") — pause/resume sets eval_paused flag; status reports last evaluated checkpoint, whether eval is running, and skipped checkpoint count
```

- [ ] **Step 4: Add evaluation to Completion Mode**

In the "Completion Mode" section, Step 1: Final Summary (around line 208), add to the summary list:

```markdown
- Evaluation results summary (from Evaluation History in experiment-context.md)
```

- [ ] **Step 5: Verify and commit**

Read `skills/watchdog/SKILL.md` and verify:
- Step 3.5 appears between step 3 and step 4 in the monitoring loop
- Async Evaluation section exists with prompt template and state variables table
- Red Flags "Always" list includes eval command handling
- Completion Mode references Evaluation History

```bash
git add skills/watchdog/SKILL.md
git commit -m "feat(watchdog): add async checkpoint evaluation via background subagent"
```

---

### Task 2: Update training-handoff experiment-context.md template

**Files:**
- Modify: `skills/training-handoff/SKILL.md:132-156` (Step 4 experiment-context.md template)

- [ ] **Step 1: Add eval_command to Training Configuration**

In the experiment-context.md template inside Step 4 (line 132-139), add the following line after `- Key hyperparameters: [lr, batch_size, etc.]` (line 139):

```markdown
- Eval command: [command with {checkpoint_path} placeholder, e.g., python eval.py --checkpoint={checkpoint_path}; leave empty if no evaluation]
```

Note: `Checkpoint directory` already exists at line 136 — no need to add it.

- [ ] **Step 2: Add Evaluation History section to template**

In the experiment-context.md template, add the following section after `## Diagnosis History` / `(empty)` (line 155) and before the closing ``` of the template:

```markdown

## Evaluation History
(populated by Watchdog during training)
```

- [ ] **Step 3: Add eval_command to Step 3 Expected checklist**

In the Step 3 "Expected" checklist (around line 62), add the following item after `- [ ] Checkpoint save with configurable interval`:

```markdown
- [ ] If evaluation is part of the experiment: eval_command defined for Watchdog async evaluation
```

- [ ] **Step 4: Verify and commit**

Read `skills/training-handoff/SKILL.md` and verify:
- Training Configuration in template has `Eval command` field
- Evaluation History section exists after Diagnosis History
- Step 3 Expected checklist has eval_command item
- All other content unchanged

```bash
git add skills/training-handoff/SKILL.md
git commit -m "feat(handoff): add eval_command and Evaluation History to experiment-context template"
```

---

### Task 3: Cross-file verification and version bump

**Files:**
- Read: both modified files
- Modify: version files

- [ ] **Step 1: Cross-check consistency**

Verify:
- Watchdog step 3.5 references `eval_command` from experiment-context.md → training-handoff template has this field
- Watchdog references `Evaluation History` section → training-handoff template has this section
- Watchdog eval subagent prompt uses `{checkpoint_path}` placeholder → matches training-handoff template description
- Watchdog references `checkpoint_dir` → training-handoff template already has `Checkpoint directory`

- [ ] **Step 2: Version bump**

Bump minor version 0.10.0 → 0.11.0 in all three version files:
- `.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`
- `.cursor-plugin/plugin.json`

- [ ] **Step 3: Commit**

```bash
git add .claude-plugin/plugin.json .claude-plugin/marketplace.json .cursor-plugin/plugin.json
git commit -m "chore: bump version to 0.11.0"
```
