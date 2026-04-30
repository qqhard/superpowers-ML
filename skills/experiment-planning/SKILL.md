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

## Revision Mode

When the orchestrator passes existing plan content AND a revised design with "## Impact on Plan" section, you are in revision mode.

<HARD-GATE>
In revision mode, you MUST edit the existing plan file in place. Do NOT create a new plan file. Preserve subtask numbering for unaffected subtasks.
</HARD-GATE>

### Flow:
1. Read existing plan fully
2. Read design's "Impact on Plan" section — which subtasks are affected
3. For each affected subtask: rewrite steps to match revised design, preserve subtask number
4. For new subtasks: append to end of plan (Task N+1, N+2, ...)
5. For removed subtasks: mark as `REMOVED: [reason]` (don't delete — human needs to see what was dropped)
6. Edit existing plan file in place
7. Commit: `"experiment: revise plan — [what changed]"`

### Marking subtask status:
Unchanged subtasks that already passed VP keep their results:
```
- [x] Task 1: ... (unchanged, VP passed)
- [ ] Task 2: ... (REVISED — needs re-execution)
- [ ] Task 5: ... (NEW)
```

### What stays the same:
- Plan Gate still applies (evaluation subtask, cadence, etc.)
- Self-review still runs
- Transitions to `spml:ml-subagent-dev` for execution of changed/new subtasks only

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

**Validation scope:** [Reference validation scope from brainstorm design doc — which levels enabled (L0/L1), data flow choice, baselines]

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

Each subtask = one functional change that can be independently tested and committed. A subtask always contains BOTH the code change AND its tests.

Split by functional boundary (model, training loop, data pipeline), NOT by artifact type (all tests, all implementation, all scripts).

**Wrong (split by artifact type):**
- Subtask 1: Rewrite model unit tests
- Subtask 2: Rewrite model implementation
- Subtask 3: Add train tests
- Subtask 4: Rewrite training script

**Right (split by functional unit, with INTEGRATION marker):**
- Subtask 1: Rewrite model (tests + implementation) — code subtask
- Subtask 2: Rewrite evaluator core (tests + implementation) — code subtask
- Subtask 3: Final Training Pipeline `[INTEGRATION]` (assembles model + evaluator + trainer + logging into runnable train.py)

If evaluation is part of the experiment, the plan should normally decompose into at least:
- model core subtask (code)
- evaluator core subtask (code, both entry modes share the same core)
- final training pipeline subtask `[INTEGRATION]` (assembles all of the above)

Trainer logic (which decides **when** evaluation fires) lives inside the integration subtask. Evaluator logic (which decides **how** evaluation runs) is its own code subtask. Do not hide evaluation inside the integration trainer code.

### Integration Subtask Rule

<HARD-GATE>
Every plan MUST contain **exactly one** subtask whose title ends with `[INTEGRATION]`. This is the subtask that assembles all components into the delivered training pipeline (typically `train.py`). The Validation Pyramid (L0 + L1) runs once on this subtask and only this subtask. All other subtasks are code subtasks and run standard superpowers reviews only.
</HARD-GATE>

If the experiment ends in a training run, the integration subtask is mandatory — `spml:ml-subagent-dev`'s Plan Gate will reject plans missing it or with multiple.

### Subtask Structure — Code Subtask (no VP)

Use this template for any subtask that is NOT marked `[INTEGRATION]`. Code subtasks are validated by TDD + Spec Review + Quality Review only. They do NOT run L0 or L1.

````markdown
## Subtask N: [Component Name]

**Role:** [What this component contributes to the integration]
**Implementation:** [What to change, which files]
**Unit Tests:** [Which custom functions need deterministic tests]
**Expected Conclusion:** "Implemented + N unit tests passing"

### Step 1: Write unit tests for custom functions

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

### Step 5: Commit

```bash
git add [specific files]
git commit -m "experiment: [subtask description]"
```
````

### Subtask Structure — Integration Subtask (single, runs VP)

Use this template for the one `[INTEGRATION]` subtask. This is the only subtask that runs the Validation Pyramid.

````markdown
## Subtask N: [Final Training Pipeline] [INTEGRATION]

**Hypothesis:** [Restate the experiment's overall hypothesis — this is what the assembled pipeline tests]
**Components consumed:** [list of upstream code subtasks the integration depends on, with file paths]
**Implementation:** [Wire components into train.py; add logging, MFU, checkpoint save/resume, fixed seeds; match production-training requirements]
**Integration Tests:** [End-to-end smoke test: data → model → loss → backward → step on a tiny shape]
**Validation Pyramid:** L0 + L1 (mandatory) — [specific metrics + baselines from brainstorm]
**Evaluation contract:** [Step-based cadence, default/full validation scope or explicit override, checkpoint-based + in-memory entry modes, observability requirements, failure-handling requirements]
**Expected Conclusion:** [What success means in terms of L1 metrics / what failure means]

### Step 1: Write integration tests

[End-to-end smoke test that exercises the full pipeline on a tiny shape.]

### Step 2: Run integration tests to verify they fail

Run: `pytest tests/path/test_integration.py -v`
Expected: FAIL (pipeline not assembled)

### Step 3: Assemble the training pipeline

[Wire components, write train.py, add production-training requirements. No test/validation imports in core code.]

### Step 4: Run integration tests to verify they pass

Run: `pytest tests/path/test_integration.py -v`
Expected: PASS

### Step 5: Write validation scripts

[Scripts that observe core code externally. Import toolkit if needed; use hooks/wrappers, never modify core.]

### Step 6: Run Validation Pyramid (L0 → L1, orchestrator-driven)

L0 (ml-static-checks): runs as a subagent — no manual command.

L1 (ml-runtime-validator):
Run: `[project-specific training command, limited to ~5 min via config/CLI flags]`
Expected: MFU >= [baseline from brainstorm], no NaN/Inf, loss decreasing, all 6 pipeline stages pass

### Step 7: Record full conclusion (with L1 metrics)

[What the results mean for the hypothesis. Include actual L1 metric values.]

### Step 8: Commit

```bash
git add [specific files]
git commit -m "experiment: integration training pipeline"
```
````

## Bite-Sized Steps Within Subtasks

Each step should be one action. Code subtasks have ~5 steps (test → fail → implement → pass → commit). Integration subtasks have ~8 (the same plus validation scripts, VP, conclusion).

## Production Training Script Requirements

If the experiment will need a long-running training phase (hours/days), include these requirements in the **`[INTEGRATION]` subtask spec** so the implementer builds them when assembling the training pipeline. Do NOT leave them for training-handoff to patch later.

**Required in the `[INTEGRATION]` subtask:**
- Human-readable log file output (one line per step with key metrics: loss, grad norm, lr)
- MFU calculation and logging
- Terminal progress indicator (tqdm or similar)
- Checkpoint save with configurable interval
- Resume from checkpoint support
- Fixed random seeds

These requirements are also what L0 (ml-static-checks) verifies on the integration subtask, so omitting them produces L0 failures.

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

**2. Parallel Session (separate)** — Open new session with superpowers:executing-plans, batch execution with checkpoints

**Which approach?"**

**If Subagent-Driven chosen:**
- **REQUIRED SUB-SKILL:** Use spml:ml-subagent-dev
- Stay in this session

**If Parallel Session chosen:**
- Guide them to open new session
- **REQUIRED SUB-SKILL:** New session uses superpowers:executing-plans
