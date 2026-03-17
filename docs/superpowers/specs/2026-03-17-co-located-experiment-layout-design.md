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

## Changes

### 1. experiment-planning (SKILL.md)

**Add a "Directory Layout Convention" section** after "Code Separation Principle", before "Plan Document Header":

- Present the canonical layout above
- State the key rule: one directory per experiment, everything inside it
- Note the top-level name is flexible

**Update "Shared Scaffold" template** to reference `[experiment-dir]` instead of generic paths.

### 2. training-handoff (SKILL.md)

**Add directory layout check** between Step 1 (Verify VP Completion) and Step 2 (Verify Training Script Readiness):

- Verify training script, tests, and outputs are under the same experiment directory
- If handoff artifacts would be written outside the training script's directory, flag to user
- **Not a hard gate** — user may have reasons for a different layout; default expectation is co-location

**Rename `[path]` → `[experiment-dir]`** throughout all templates:

- Step 3: `[experiment-dir]/experiment-context.md`
- Step 4: `[experiment-dir]/watchdog-prompt.md`, log at `[experiment-dir]/outputs/train.log`
- Step 5: All paths in launch instructions use `[experiment-dir]`

### 3. No changes needed

- **watchdog** — Already reads paths from experiment-context.md; will automatically pick up the new paths
- **training-resume** — Same; reads experiment-context.md
- **verification** — No path dependencies
- **subagent-dev** — Delegates to training-handoff; no direct path logic
