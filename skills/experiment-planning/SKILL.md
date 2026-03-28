---
name: experiment-planning
description: Use when you have an ML experiment design or requirements for a multi-step ML task, before touching code
---

# ML Experiment Planning

## Overview

Write comprehensive ML experiment plans assuming the engineer has zero context for the codebase and limited ML debugging experience. Document everything: which files to touch, what to implement, what to test, how to validate, what the expected outcomes are. Break into atomic subtasks. YAGNI. Code separation. Frequent commits.

Assume the implementer is a skilled developer but may not recognize when ML code "runs but is wrong."

**Announce at start:** "I'm using the experiment-planning skill to create the implementation plan."

**Save plans to:** `<experiment_dir>/plans/YYYY-MM-DD-<experiment-name>.md` (use the experiment directory from the brainstorm design doc)

## Code Separation Principle

**CRITICAL:** Core code (model, training, data) must never import from test/validation code or toolkit. Validation scripts observe core code externally via hooks/wrappers. After development, core code can be extracted and deployed to production as-is.

The agent determines where to place test and validation code based on the user's existing project structure.

## Directory Layout Convention

Each experiment lives in a single directory. All artifacts — core code, tests, training outputs, and handoff files — are co-located under this directory. This makes experiments self-contained and easy to find.

```
[experiment-dir]/
├── plans/
│   ├── YYYY-MM-DD-<topic>-design.md       # brainstorm design doc
│   └── YYYY-MM-DD-<experiment-name>.md    # implementation plan
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

## Plan Document Header

**Every plan MUST start with this header:**

```markdown
# [Experiment Name] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** [One sentence]

**Experiment directory:** [Path from brainstorm design doc, e.g. `experiments/my-experiment/`]

**Hypothesis:** [Doing X is expected to cause Y] (if applicable)

**Validation scope:** [Reference validation scope from brainstorm design doc — which levels enabled (L0/L1/L2), data flow choice, baselines, L2 step count]

**Evaluation design:** [Whether evaluation is required, step-based cadence, default/full validation scope or explicit override, both entry modes, observability requirements, failure-handling expectations]

**Architecture:** [2-3 sentences about approach]

---
```

## Plan Structure

Plans have two sections: shared scaffold, then atomic subtasks.

### Shared Scaffold Section

```markdown
## Shared Scaffold

### Existing infra (don't touch, advise if problems found)
- Data pipeline: `path/to/data_loader.py`
- Training loop: `path/to/trainer.py`
- [list all existing infra identified in brainstorm]

### Needs setup
- Experiment directory: `[experiment-dir]/`
- [Only what's missing, with exact file paths and implementation]
```

### Subtask Decomposition

Each subtask = one functional change that can be independently tested and committed. A subtask always contains BOTH the code change AND its tests (Steps 1-4 in the template below).

Split by functional boundary (model, training loop, data pipeline), NOT by artifact type (all tests, all implementation, all scripts).

**Wrong (split by artifact type):**
- Subtask 1: Rewrite model unit tests
- Subtask 2: Rewrite model implementation
- Subtask 3: Add train tests
- Subtask 4: Rewrite training script

**Right (split by functional unit):**
- Subtask 1: Rewrite model (tests + implementation)
- Subtask 2: Rewrite training script (tests + implementation)

If evaluation is part of the experiment, the plan should normally decompose into at least:
- model core subtask
- trainer subtask
- evaluation subtask

The evaluation subtask must cover both entry modes through one shared evaluator core:
- standalone checkpoint-based evaluation
- in-training evaluation from in-memory state

Do not hide evaluation inside a generic trainer subtask. Trainer owns **when** evaluation fires. Evaluator owns **how** evaluation runs.

### Subtask Structure

Each subtask goes through the Validation Pyramid.

````markdown
## Subtask N: [Experiment/Component Name]

**Hypothesis:** [Specific hypothesis for this subtask]
**Implementation:** [What to change, which files]
**Unit Tests:** [Which custom functions need traditional deterministic tests]
**Validation Pyramid:** [Which levels apply (L0/L1/L2) + specific metrics + baselines from brainstorm]
**Evaluation contract:** [Step-based cadence, default/full validation scope or explicit override, checkpoint-based + in-memory entry modes, observability requirements, failure-handling requirements]
**Expected Conclusion:** [What success means / what failure means]

### Step 1: Write unit tests for custom functions

[Only for deterministic code: custom loss, custom layers, data transforms]

```python
def test_custom_loss_basic():
    pred = torch.tensor([0.5, 0.3, 0.2])
    target = torch.tensor([1.0, 0.0, 0.0])
    loss = custom_loss(pred, target)
    assert loss.shape == ()
    assert not torch.isnan(loss)
```

### Step 2: Run unit tests to verify they fail

Run: `pytest tests/path/test_custom_loss.py -v`
Expected: FAIL (function not defined)

### Step 3: Implement core code

[Exact code, exact file paths. This code goes in the core/src directory — no test/validation imports.]

### Step 4: Run unit tests to verify they pass

Run: `pytest tests/path/test_custom_loss.py -v`
Expected: PASS

### Step 5: Write validation scripts

[Scripts that observe core code externally. These import from toolkit if needed and use hooks/wrappers.]

### Step 6: Run Validation Pyramid

L0 (static analysis) is handled by code review — no separate command needed.

L1 (runtime validation):
Run: `[project-specific training command with monitoring]`
Expected: MFU >= [baseline from brainstorm], no NaN/Inf, loss decreasing

L2 (E2E pipeline):
Run: `[project-specific E2E validation command]`
Expected: All 6 stages complete without error

### Step 7: Record conclusion

[What the results mean for the hypothesis]

### Step 8: Commit

```bash
git add [specific files]
git commit -m "experiment: [subtask description]"
```
````

## Bite-Sized Steps Within Subtasks

Each step should be one action:
- "Write the unit test" — step
- "Run it to make sure it fails" — step
- "Write the core implementation" — step
- "Run unit tests to verify" — step
- "Write validation scripts" — step
- "Run Validation Pyramid" — step
- "Record conclusion" — step
- "Commit" — step

## Production Training Script Requirements

If the experiment includes a training script subtask AND will need a long-running training phase (hours/days), include these requirements in the subtask spec so the implementer builds them from the start. Do NOT leave them for training-handoff to patch later.

**Required in the training script subtask:**
- Human-readable log file output (one line per step with key metrics: loss, grad norm, lr)
- MFU calculation and logging
- Terminal progress indicator (tqdm or similar)
- Checkpoint save with configurable interval
- Resume from checkpoint support
- Fixed random seeds

## Evaluation Planning Requirements

If the experiment has a validation or evaluation phase, plans are incomplete unless they explicitly state:

- step-based evaluation cadence, e.g. `every 500 steps`
- evaluation scope, defaulting to `full validation` unless explicitly overridden
- both evaluation entry modes:
  - checkpoint-based
  - in-memory during training
- one shared evaluator core across both modes
- evaluation observability requirements:
  - phase-start message
  - dedicated progress bar
  - phase-end message
  - result summary
  - efficiency summary
- evaluation failure-handling expectations:
  - checkpoint missing/unreadable
  - checkpoint restore failure
  - empty or misconfigured validation dataloader
  - metric aggregation failure
  - non-finite metrics
  - long silent gaps or stalled evaluation

Final-epoch-only evaluation is not an acceptable default for long-running training.

These enable the Watchdog to monitor training and the user to track progress. Without them, training-handoff will flag gaps and potentially need to modify VP-validated code.

## Remember
- Exact file paths always
- Complete code in plan (not "add validation")
- Exact commands with expected output ranges
- Core code never imports from test/validation
- Validation observes externally via hooks/wrappers
- YAGNI, code separation, frequent commits

## Execution Handoff

After saving the plan, offer execution choice:

**"Plan complete and saved to `<experiment_dir>/plans/<filename>.md`. Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per subtask, review between subtasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

**Which approach?"**

**If Subagent-Driven chosen:**
- **REQUIRED SUB-SKILL:** Use spml:ml-subagent-dev
- Stay in this session

**If Parallel Session chosen:**
- Guide them to open new session
- **REQUIRED SUB-SKILL:** New session uses spml:executing-plans
