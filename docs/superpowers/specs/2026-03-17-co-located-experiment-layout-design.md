# Co-located Experiment Directory Layout — Design Spec

## Problem

Handoff artifacts (experiment-context.md, watchdog-prompt.md), training scripts, tests, and outputs can end up scattered across unrelated directories (e.g., `scripts/`, `docs/`, `outputs/`, `tests/`). This makes experiments hard to navigate and maintain.

## Goal

Establish a directory layout convention in SPML skills so that each experiment is self-contained in a single directory.

## Approach

**Option B (chosen):** Define the convention in `experiment-planning`, reinforce with a check in `training-handoff`.

## Canonical Layout

```
experiments/<experiment-name>/
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

The top-level directory name (`experiments/`) is a default — use whatever fits the user's existing project structure. The key rule: **one directory per experiment, everything inside it.**

**Scope:** This convention applies to new experiments only. Existing experiments are not migrated.

**Plan documents** remain in `docs/plans/` — they are planning artifacts, not experiment runtime artifacts.

## Changes

### 1. experiment-planning (SKILL.md)

**Add a "Directory Layout Convention" section** after "Code Separation Principle", before "Plan Document Header":

- Present the canonical layout above
- State the key rule: one directory per experiment, everything inside it
- Note the top-level name is flexible

**Add `[experiment-dir]` to "Shared Scaffold" template** — add a line establishing the experiment directory path (e.g., `- Experiment directory: [experiment-dir]/`). Existing infra paths (e.g., `path/to/data_loader.py`) stay as-is since existing infra may live outside the experiment directory.

### 2. training-handoff (SKILL.md)

**Add directory layout check** between Step 1 (Verify VP Completion) and Step 2 (Verify Training Script Readiness):

- Verify training script, tests, and outputs are under the same experiment directory
- If handoff artifacts would be written outside the training script's directory, flag to user
- **Not a hard gate** — user may have reasons for a different layout; default expectation is co-location

**Rename `[path]` → `[experiment-dir]`** in all occurrences across Steps 3-5:

- Step 3 (experiment-context.md template):
  - `Script: [experiment-dir]/train.py` (or actual script name)
  - `Log file: [experiment-dir]/outputs/train.log`
  - `Checkpoint directory: [experiment-dir]/outputs/`
- Step 4 (watchdog-prompt.md template):
  - `Read [experiment-dir]/experiment-context.md`
  - `Locate the training log at [experiment-dir]/outputs/train.log`
  - **Note:** This changes the subdirectory from `logs/` to `outputs/` and the filename from `training.log` to `train.log` to match the canonical layout
- Step 5 (launch instructions):
  - All paths use `[experiment-dir]` prefix

### 3. watchdog (SKILL.md) — cosmetic only

Rename `[path]` → `[experiment-dir]` in the completion-prompt.md template for consistency. (recovery-prompt.md is referenced in prose but has no template block in the current skill — no change needed there.) No behavioral change — watchdog reads actual paths from experiment-context.md at runtime.

### 4. No changes needed

- **training-resume** — Reads paths from experiment-context.md; picks up new paths automatically
- **verification** — No path dependencies
- **subagent-dev** — Delegates to training-handoff; no direct path logic
