# Step-Based Evaluation Subtask Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update SPML planning, implementation, and validation guidance so evaluation is treated as a first-class subtask with step-based cadence, dual entry modes, and mandatory progress visibility.

**Architecture:** Propagate the new evaluation model through the existing planning and execution guidance rather than adding a new standalone skill. The main changes land in `brainstorming`, `subagent-dev`, and the validation/handoff documentation that checks or depends on training-time evaluation behavior.

**Tech Stack:** Markdown skill definitions and documentation

**Spec:** `docs/superpowers/specs/2026-03-18-step-based-evaluation-subtask-design.md`

---

### Task 1: Update brainstorming guidance to require evaluation structure upstream

**Files:**
- Modify: `skills/brainstorming/SKILL.md`
- Test: review rendered markdown content manually

- [ ] **Step 1: Read the current brainstorming skill**

Run: `sed -n '1,260p' skills/brainstorming/SKILL.md`
Expected: Current ML brainstorming flow, including validation-scope questions and planning transition

- [ ] **Step 2: Add evaluation-specific context collection requirements**

Update `skills/brainstorming/SKILL.md` so the brainstorming flow explicitly asks about:

- whether evaluation is required
- step-based evaluation cadence (`every N steps`)
- evaluation scope, with default `full validation` unless explicitly overridden
- whether both entry modes are needed:
  - from checkpoint
  - from in-memory training state
- progress visibility expectations for long-running evaluation
- evaluation efficiency expectations (time-to-first-progress, throughput, checkpoint load behavior)
- evaluation failure expectations:
  - checkpoint missing/unreadable
  - checkpoint restore failure
  - empty or misconfigured validation dataloader
  - metric aggregation failure
  - non-finite metrics
  - stalled evaluation / no progress output

The new text should make these decisions part of experiment design, not something deferred to handoff.

- [ ] **Step 3: Add evaluation requirements to the validation-scope/design sections**

In the sections that currently describe validation scope and design coverage, add explicit guidance that:

- evaluation is a dedicated subtask
- trainer owns trigger timing only
- evaluator owns execution behavior
- plans must record `full validation` as the default scope unless they explicitly override it
- final-epoch-only validation is not an acceptable default
- failure handling is part of design completeness, not an implementation afterthought

- [ ] **Step 4: Review the updated file for consistency**

Run: `sed -n '1,260p' skills/brainstorming/SKILL.md`
Expected: New evaluation guidance appears once, in the correct upstream sections, without contradicting existing VP flow

- [ ] **Step 5: Commit**

```bash
git add skills/brainstorming/SKILL.md
git commit -m "docs: require evaluation design during brainstorming"
```

### Task 2: Update experiment planning to make evaluation a first-class subtask

**Files:**
- Modify: `skills/subagent-dev/SKILL.md`
- Modify: `skills/using-superpowers-ml/SKILL.md`
- Test: review markdown content manually

- [ ] **Step 1: Read the current execution/planning-adjacent skills**

Run: `sed -n '1,260p' skills/subagent-dev/SKILL.md`
Expected: Current subagent execution workflow and validation pyramid references

Run: `sed -n '1,220p' skills/using-superpowers-ml/SKILL.md`
Expected: Current ML skill selection and priority guidance

- [ ] **Step 2: Add explicit planning requirements for evaluation subtasks**

Update the relevant sections of `skills/subagent-dev/SKILL.md` so plans and task decomposition require:

- separate `model core`, `trainer`, and `evaluation` subtasks
- evaluation subtask coverage for both:
  - standalone checkpoint evaluation
  - in-training evaluation from in-memory state
- one shared evaluator core across both entry modes
- step-based evaluation cadence as the default plan expression
- `full validation` as the default evaluation scope unless the plan states an override
- explicit failure-handling requirements for the evaluation boundary

The updated review/decomposition language must say incomplete plans fail review if they:

- omit the evaluation subtask
- omit step-based cadence
- omit default/full evaluation scope or its explicit override
- omit either required entry mode
- omit evaluation progress visibility requirements
- omit mode-aware failure handling requirements

The wording should make clear that evaluation is not a tail step hidden inside trainer implementation.

- [ ] **Step 3: Tighten top-level ML workflow guidance**

Update `skills/using-superpowers-ml/SKILL.md` to mention that training/evaluation work should be structured so evaluation is observable, dual-entry, and planned explicitly upstream when validation/evaluation is part of the task.

- [ ] **Step 4: Review the updated files**

Run: `sed -n '1,320p' skills/subagent-dev/SKILL.md`
Expected: Evaluation subtask requirements are present and aligned with the spec

Run: `sed -n '1,240p' skills/using-superpowers-ml/SKILL.md`
Expected: High-level guidance does not contradict the new evaluation structure

- [ ] **Step 5: Commit**

```bash
git add skills/subagent-dev/SKILL.md skills/using-superpowers-ml/SKILL.md
git commit -m "docs: make evaluation a first-class ML subtask"
```

### Task 3: Update validation and handoff expectations for evaluation observability

**Files:**
- Modify: `skills/training-handoff/SKILL.md`
- Modify: `docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md`
- Test: review markdown content manually

- [ ] **Step 1: Read current handoff and observability docs**

Run: `sed -n '1,260p' skills/training-handoff/SKILL.md`
Expected: Current production-readiness checks including progress indicator expectations

Run: `sed -n '1,240p' docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md`
Expected: Existing training logging and progress-bar checks

- [ ] **Step 2: Extend handoff readiness checks to cover evaluation**

Update `skills/training-handoff/SKILL.md` so production-readiness checks explicitly verify:

- evaluation exists as a distinct capability, not only a final-epoch block
- both evaluation entry modes are available when evaluation is part of the experiment
- long-running evaluation has visible phase messages and a dedicated progress bar
- evaluation emits a summary that includes both metrics and efficiency/latency information
- evaluation surfaces mode-aware status and error context:
  - checkpoint mode reports checkpoint path/load behavior
  - in-training mode reports that evaluation used in-memory state
- evaluation readiness checks include explicit failure modes:
  - checkpoint missing/unreadable
  - checkpoint restore failure
  - empty/misconfigured validation dataloader
  - metric aggregation failure
  - non-finite metrics
  - long silent evaluation gaps

Keep the handoff role limited to verification and gap-reporting rather than rewriting the whole training script.

- [ ] **Step 3: Extend observability spec language to cover evaluation phase**

Update `docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md` to clarify that the observability expectations apply not only to training but also to evaluation, especially:

- explicit evaluation phase start/end markers
- evaluation progress output
- evaluation timing/throughput summaries
- time-to-first-progress-update checks
- checkpoint load latency checks for checkpoint-triggered evaluation
- explicit reporting of evaluation mode (checkpoint-based vs in-memory)
- cadence-firing checks for in-training evaluation
- no long silent gaps during evaluation

Do not redesign the old spec; make a focused update that keeps it aligned with the new evaluation-subtask design.

- [ ] **Step 4: Review both files**

Run: `sed -n '1,320p' skills/training-handoff/SKILL.md`
Expected: Handoff now checks for evaluation observability and dual entry modes

Run: `sed -n '1,260p' docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md`
Expected: Observability spec clearly mentions evaluation-phase visibility requirements

- [ ] **Step 5: Commit**

```bash
git add skills/training-handoff/SKILL.md docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md
git commit -m "docs: extend observability guidance to evaluation"
```

### Task 4: Add plan-review and runtime-enforcement references

**Files:**
- Modify: `skills/subagent-dev/SKILL.md`
- Modify: `skills/training-handoff/SKILL.md`
- Test: review markdown content manually

- [ ] **Step 1: Read the current downstream enforcement sections**

Run: `sed -n '1,360p' skills/subagent-dev/SKILL.md`
Expected: Current execution workflow and any existing plan/review requirements

Run: `sed -n '1,320p' skills/training-handoff/SKILL.md`
Expected: Current handoff checks and launch guidance

- [ ] **Step 2: Add explicit runtime-enforcement language**

Update the relevant sections so downstream enforcement consistently expects:

- in-training evaluation actually fires at the planned step cadence
- checkpoint-based evaluation reports checkpoint load behavior
- in-training evaluation reports that it is using in-memory state
- evaluation start/end messages and progress output are treated as runtime checks, not stylistic suggestions
- errors surface with mode-aware context at the evaluation boundary

This task should update review/check language, not add a new workflow stage.

- [ ] **Step 3: Re-read the updated downstream docs**

Run: `sed -n '1,360p' skills/subagent-dev/SKILL.md`
Expected: Subagent workflow now reflects evaluation-specific enforcement points

Run: `sed -n '1,320p' skills/training-handoff/SKILL.md`
Expected: Handoff/runtime checks now mention cadence firing, mode reporting, and mode-aware errors

- [ ] **Step 4: Commit**

```bash
git add skills/subagent-dev/SKILL.md skills/training-handoff/SKILL.md
git commit -m "docs: add evaluation runtime enforcement guidance"
```

### Task 5: Verify the updated documentation set end to end

**Files:**
- Test: `skills/brainstorming/SKILL.md`
- Test: `skills/subagent-dev/SKILL.md`
- Test: `skills/using-superpowers-ml/SKILL.md`
- Test: `skills/training-handoff/SKILL.md`
- Test: `docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md`

- [ ] **Step 1: Search for evaluation-related guidance**

Run: `rg -n "evaluation|validate|validation cadence|checkpoint-based|in-memory|progress bar" skills docs/superpowers/specs`
Expected: Updated files contain consistent language about evaluation subtask structure, dual entry modes, and progress visibility

- [ ] **Step 2: Manually verify no contradictory guidance remains**

Review the search output and confirm there is no surviving guidance that:

- defaults to final-only evaluation
- treats evaluation as only a trainer tail step
- drops the default `full validation` scope without an explicit override
- allows only one evaluation entry mode when both are required
- tolerates long-running evaluation without visible progress
- omits mode-aware failure handling expectations
- omits runtime checks for cadence firing or evaluation mode reporting

- [ ] **Step 3: Check git diff for scope control**

Run: `git diff -- skills/brainstorming/SKILL.md skills/subagent-dev/SKILL.md skills/using-superpowers-ml/SKILL.md skills/training-handoff/SKILL.md docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md`
Expected: Diff is limited to evaluation-subtask and observability alignment changes

- [ ] **Step 4: Final commit**

```bash
git add skills/brainstorming/SKILL.md skills/subagent-dev/SKILL.md skills/using-superpowers-ml/SKILL.md skills/training-handoff/SKILL.md docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md
git commit -m "docs: align SPML workflow around evaluation subtasks"
```
