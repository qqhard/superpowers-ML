# Co-located Experiment Directory Layout — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a co-located directory layout convention to SPML skills so experiments are self-contained.

**Architecture:** Add a "Directory Layout Convention" section to experiment-planning, add a layout check + rename `[path]` → `[experiment-dir]` in training-handoff, and do a cosmetic rename in watchdog.

**Spec:** `docs/superpowers/specs/2026-03-17-co-located-experiment-layout-design.md`

---

## File Structure

- Modify: `skills/experiment-planning/SKILL.md` — add layout convention section + scaffold line
- Modify: `skills/training-handoff/SKILL.md` — add layout check + rename all `[path]` placeholders
- Modify: `skills/watchdog/SKILL.md` — cosmetic rename of one `[path]` placeholder

---

### Task 1: Add Directory Layout Convention to experiment-planning

**Files:**
- Modify: `skills/experiment-planning/SKILL.md:18-22` (after Code Separation Principle)
- Modify: `skills/experiment-planning/SKILL.md:50-60` (Shared Scaffold template)

- [ ] **Step 1: Add "Directory Layout Convention" section**

Insert after line 22 (end of Code Separation Principle section), before "## Plan Document Header":

```markdown
## Directory Layout Convention

Each experiment lives in a single directory. All artifacts — core code, tests, training outputs, and handoff files — are co-located under this directory. This makes experiments self-contained and easy to find.

```
[experiment-dir]/
├── train.py                  # training script
├── model.py                  # model code (if separate)
├── data.py                   # data loading (if separate)
├── experiment-context.md     # handoff artifact
├── watchdog-prompt.md        # handoff artifact
├── outputs/
│   ├── train.log             # training log
│   └── ckpt.pt              # checkpoint(s)
└── tests/
    ├── test_model.py         # unit tests
    └── test_train.py         # validation scripts
```

The top-level directory name is flexible — use whatever fits the user's existing project structure (e.g., `experiments/seq_tower/`). The key rule: **one directory per experiment, everything inside it.**
```

- [ ] **Step 2: Add experiment directory line to Shared Scaffold template**

In the Shared Scaffold template (line 50-60), add an `Experiment directory` line under `### Needs setup`:

```markdown
### Needs setup
- Experiment directory: `[experiment-dir]/`
- [Only what's missing, with exact file paths and implementation]
```

Existing infra paths (`path/to/data_loader.py`, etc.) stay as-is — existing infra may live outside the experiment directory.

- [ ] **Step 3: Verify the edit**

Read back `skills/experiment-planning/SKILL.md` and confirm:
1. "Directory Layout Convention" section appears between "Code Separation Principle" and "Plan Document Header"
2. Shared Scaffold template has the `Experiment directory` line
3. No other sections were accidentally modified

- [ ] **Step 4: Commit**

```bash
git add skills/experiment-planning/SKILL.md
git commit -m "feat(experiment-planning): add co-located directory layout convention"
```

---

### Task 2: Add layout check and rename placeholders in training-handoff

**Files:**
- Modify: `skills/training-handoff/SKILL.md`

- [ ] **Step 1: Update checklist to include layout check**

Change the checklist (lines 29-33) to insert the new step:

```markdown
## Checklist

1. **Verify VP completion** — all enabled layers passed with actual numbers
2. **Verify directory layout** — experiment artifacts are co-located
3. **Verify training script readiness** — check existing script against production requirements
4. **Write experiment-context.md** — full context for downstream sessions
5. **Write watchdog-prompt.md** — prompt for Watchdog session
6. **Present launch instructions** — how to start the Watchdog session
```

- [ ] **Step 2: Add "Step 2: Verify Directory Layout" section**

Insert between current Step 1 and current Step 2 (which becomes Step 3):

```markdown
## Step 2: Verify Directory Layout

Check that the experiment follows the co-located layout convention from `spml:experiment-planning`:

- Training script, tests, and outputs are under the same experiment directory
- Handoff artifacts (experiment-context.md, watchdog-prompt.md) will be written to that same directory

**Not a hard gate** — the user may have reasons for a different layout. If artifacts would be written outside the training script's directory, flag this to the user and ask whether to proceed or reorganize.
```

- [ ] **Step 3: Renumber existing Steps 2-5 → Steps 3-6**

Update all step headings:
- "Step 2" → "Step 3"
- "Step 3" → "Step 4"
- "Step 4" → "Step 5"
- "Step 5" → "Step 6"

- [ ] **Step 4: Rename `[path]` → `[experiment-dir]` in Step 4 (experiment-context.md template)**

Three replacements in the Training Configuration section:

| Before | After |
|--------|-------|
| `- Script: [path to run_training.sh or train.py command]` | `- Script: [experiment-dir]/train.py (or actual script name)` |
| `- Log file: [path to training.log]` | `- Log file: [experiment-dir]/outputs/train.log` |
| `- Checkpoint directory: [path]` | `- Checkpoint directory: [experiment-dir]/outputs/` |

- [ ] **Step 5: Rename `[path]` → `[experiment-dir]` in Step 5 (watchdog-prompt.md template)**

Two replacements (these are substrings within longer lines — replace the matched portion):

| Before | After |
|--------|-------|
| `` `[path]/experiment-context.md` for full experiment context, VP baseline, and watchdog mode `` | `` `[experiment-dir]/experiment-context.md` for full experiment context, VP baseline, and watchdog mode `` |
| `` the training log at `[path]/logs/training.log` `` | `` the training log at `[experiment-dir]/outputs/train.log` `` |

Note: subdirectory changes from `logs/` to `outputs/`, filename from `training.log` to `train.log`.

- [ ] **Step 6: Rename `[path]` → `[experiment-dir]` in Step 6 (launch instructions)**

Replace all four `[path]` occurrences:

```
Handoff complete. All artifacts generated:
- Training script: [experiment-dir]/train.py
- Log file: [experiment-dir]/outputs/train.log (human-readable, one line per step)
- Experiment context: [experiment-dir]/experiment-context.md
- Watchdog prompt: [experiment-dir]/watchdog-prompt.md
- Watchdog mode: [mode] (configurable in experiment-context.md)

To start:
  1. Open a new agent session
  2. Paste the contents of watchdog-prompt.md
  3. The Watchdog will launch training and begin monitoring
```

- [ ] **Step 7: Verify the edit**

Read back `skills/training-handoff/SKILL.md` and confirm:
1. Checklist has 6 items with "Verify directory layout" as item 2
2. New Step 2 section exists between Step 1 and Step 3
3. Zero remaining `[path]` occurrences (search for it)
4. All `[experiment-dir]` placeholders are consistent

- [ ] **Step 8: Commit**

```bash
git add skills/training-handoff/SKILL.md
git commit -m "feat(training-handoff): add layout check, rename [path] to [experiment-dir]"
```

---

### Task 3: Cosmetic rename in watchdog

**Files:**
- Modify: `skills/watchdog/SKILL.md:231`

- [ ] **Step 1: Rename `[path]` → `[experiment-dir]` in completion-prompt.md template**

Line 231, change:
```
Read `[path]/experiment-context.md` for the full context including:
```
to:
```
Read `[experiment-dir]/experiment-context.md` for the full context including:
```

- [ ] **Step 2: Verify no other `[path]` occurrences remain**

Search `skills/watchdog/SKILL.md` for `[path]` — should return zero results.

- [ ] **Step 3: Commit**

```bash
git add skills/watchdog/SKILL.md
git commit -m "chore(watchdog): rename [path] to [experiment-dir] for consistency"
```
