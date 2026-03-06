---
name: ml-experiment-planning
description: Use when you have an ML experiment design or requirements for a multi-step ML task, before touching code
---

# ML Experiment Planning

## Overview

Write comprehensive ML experiment plans assuming the engineer has zero context for the codebase and limited ML debugging experience. Document everything: which files to touch, what to implement, what to test, how to validate, what the expected outcomes are. Break into atomic subtasks. YAGNI. Code separation. Frequent commits.

Assume the implementer is a skilled developer but may not recognize when ML code "runs but is wrong."

**Announce at start:** "I'm using the ml-experiment-planning skill to create the implementation plan."

**Save plans to:** `docs/plans/YYYY-MM-DD-<experiment-name>.md`

## Code Separation Principle

**CRITICAL:** Core code (model, training, data) must never import from test/validation code or toolkit. Validation scripts observe core code externally via hooks/wrappers. After development, core code can be extracted and deployed to production as-is.

The agent determines where to place test and validation code based on the user's existing project structure.

## Plan Document Header

**Every plan MUST start with this header:**

```markdown
# [Experiment Name] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-ml:ml-subagent-dev to implement this plan task-by-task. (If ml-subagent-dev is not yet available, use superpowers-ml:executing-plans.)

**Goal:** [One sentence]

**Hypothesis:** [Doing X is expected to cause Y] (if applicable)

**Validation scope:** [Reference validation scope from brainstorm design doc — which layers enabled, which skipped, key thresholds]

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
- [Only what's missing, with exact file paths and implementation]
```

### Subtask Structure

Each subtask is an atomic experiment or implementation unit. Each goes through the Validation Pyramid.

````markdown
## Subtask N: [Experiment/Component Name]

**Hypothesis:** [Specific hypothesis for this subtask]
**Implementation:** [What to change, which files]
**Unit Tests:** [Which custom functions need traditional deterministic tests]
**Validation Pyramid:** [Which layers apply + specific metrics + thresholds from brainstorm]
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

[Exact commands for each enabled layer, with expected output ranges]

Run: `python validation/run_l0_efficiency.py`
Expected: MFU >= 0.40, no backend warnings

Run: `python validation/run_l1_process_metrics.py`
Expected: No NaN/Inf, gradient norm in [1e-4, 1e2], loss decreasing

Run: `python validation/run_l2_overfit.py`
Expected: Loss < 0.01 after 10 epochs on 100 samples

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

## Remember
- Exact file paths always
- Complete code in plan (not "add validation")
- Exact commands with expected output ranges
- Core code never imports from test/validation
- Validation observes externally via hooks/wrappers
- YAGNI, code separation, frequent commits

## Execution Handoff

After saving the plan, offer execution choice:

**"Plan complete and saved to `docs/plans/<filename>.md`. Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per subtask, review between subtasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

**Which approach?"**

**If Subagent-Driven chosen:**
- **REQUIRED SUB-SKILL:** Use superpowers-ml:ml-subagent-dev (or superpowers-ml:subagent-driven-development if ml-subagent-dev not yet available)
- Stay in this session

**If Parallel Session chosen:**
- Guide them to open new session
- **REQUIRED SUB-SKILL:** New session uses superpowers-ml:executing-plans
