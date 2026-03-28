---
name: ml-subagent-dev
description: Use when executing ML experiment plans with subagents - adapts superpowers:subagent-driven-development with 3-level Validation Pyramid (L0 static → L1 runtime → L2 E2E), experiment-aware reviews, and conclusion recording
---

<HARD-GATE>
Every subtask MUST complete ALL of the following before it can be marked complete.
No exceptions. No "this is too simple". No "this is just a toy experiment".

A subtask without VP results and Review results is NOT complete. Period.

## Subtask Completion Gate

Before marking ANY subtask as complete, you MUST have:

- [ ] L0: VP Static Checks — passed (with actual numbers recorded)
- [ ] L1: ML Runtime Validator — passed (with actual metrics recorded)
- [ ] L2: ML E2E Validator — passed (with actual pipeline stages confirmed)
- [ ] Spec Review — passed (experiment design compliance confirmed)
- [ ] Quality Review — passed (code quality confirmed)
- [ ] Conclusion recorded — with metric evidence from VP

If ANY item is unchecked, the subtask is NOT complete.
Do NOT proceed to the next subtask. Do NOT mark it as done.
</HARD-GATE>

## Anti-Pattern: "This Subtask Doesn't Need Full VP"

This is the single most dangerous rationalization in ML experiments.

| Thought | Reality |
|---------|---------|
| "This is just a toy experiment" | Toy experiments with wrong gradients waste days of debugging |
| "The model code is simple" | Simple code with silent shape bugs produces plausible but wrong results |
| "Unit tests already passed" | Unit tests check deterministic logic. VP checks training dynamics. They test different things. |
| "L1/L2 is overkill for this subtask" | If this subtask is part of an ML experiment, it WILL be trained and evaluated. VP validates that. |
| "I'll run VP at the end" | VP per subtask catches bugs early. VP at the end means debugging the entire codebase at once. |
| "The user wants speed" | Skipping VP and debugging silent failures later is SLOWER. |

ML experiments ALWAYS involve a training-evaluation pipeline. Even outputting "loss is decreasing" is evaluation. If there is a pipeline, there is a VP. No exceptions.

# ML Subagent-Driven Development

Execute ML experiment plans by dispatching fresh subagent per subtask, with ML-adapted review criteria: spec compliance (does it match experiment design?), quality review (did Validation Pyramid pass?), and conclusion recording.

**Core principle:** Fresh subagent per subtask + experiment-aware review + conclusion recording = correct implementations with trustworthy conclusions.

**Adapted from:** `superpowers:subagent-driven-development`. Key changes:
- TDD = Validation Pyramid: unit tests → implement → L0 → L1 → L2, THEN Spec Review → Quality Review
- L0: VP Static Checks subagent checks static ML correctness
- L1: Runtime validation (minutes-level training run with metrics collection)
- L2: E2E pipeline validation (1-5 steps per stage)
- Spec reviewer checks experiment design compliance (hypothesis, variable control)
- Quality reviewer checks code quality (VP results already validated by orchestrator)
- Each subtask records: metric data, conclusion, anomaly log
- Shared fix loop: fail → Implementer fixes → re-run level → 5 failures → user intervention
- Large fix rollback: fix > 50 lines → re-run all prior reviews + VP levels

## When to Use

- You have an ML experiment plan (from experiment-planning)
- Subtasks are mostly independent
- You want to stay in this session (vs. executing-plans in parallel session)

## Plan Gate

Before dispatching any implementer subagent, read the plan and fail fast if a training task with validation/evaluation is missing any of the following:

- a dedicated evaluation subtask
- step-based evaluation cadence
- evaluation scope, defaulting to `full validation` unless explicitly overridden
- both required evaluation entry modes:
  - checkpoint-based
  - in-memory during training
- one shared evaluator core across both entry modes
- evaluation progress visibility requirements
- mode-aware failure-handling requirements at the evaluation boundary
- runtime checks for cadence firing and evaluation mode reporting

Do not treat these as advisory. Incomplete plans must be sent back for revision before implementation starts.

## The Process

```dot
digraph process {
    rankdir=TB;

    subgraph cluster_per_subtask {
        label="Per Subtask";
        "Dispatch ML implementer subagent" [shape=box];
        "Implementer: unit tests + implement" [shape=box];
        "Dispatch ML spec reviewer" [shape=box];
        "Spec reviewer: experiment design compliance?" [shape=diamond];
        "Implementer fixes spec gaps" [shape=box];
        "Dispatch ML quality reviewer" [shape=box];
        "Quality reviewer: code quality?" [shape=diamond];
        "Implementer fixes quality issues" [shape=box];
        "L0: VP Static Checks" [shape=box style=filled fillcolor=lightyellow];
        "L0 passed?" [shape=diamond];
        "Implementer fixes L0 issues" [shape=box];
        "L1: ML Runtime Validator" [shape=box style=filled fillcolor=lightyellow];
        "L1 passed?" [shape=diamond];
        "Implementer fixes L1 issues" [shape=box];
        "L2: ML E2E Validator" [shape=box style=filled fillcolor=lightyellow];
        "L2 passed?" [shape=diamond];
        "Implementer fixes L2 issues" [shape=box];
        "Record conclusion" [shape=box style=filled fillcolor=lightgreen];
    }

    "Read plan, extract subtasks, create tracker\n(TodoWrite / update_plan)" [shape=box];
    "More subtasks?" [shape=diamond];
    "Needs long-running training?" [shape=diamond style=filled fillcolor=lightyellow];
    "Invoke spml:training-handoff" [shape=box style=filled fillcolor=orange];
    "Invoke spml:verification" [shape=box style=filled fillcolor=lightblue];

    "Read plan, extract subtasks, create tracker\n(TodoWrite / update_plan)" -> "Dispatch ML implementer subagent";
    "Dispatch ML implementer subagent" -> "Implementer: unit tests + implement";
    "Implementer: unit tests + implement" -> "Dispatch ML spec reviewer";
    "Dispatch ML spec reviewer" -> "Spec reviewer: experiment design compliance?";
    "Spec reviewer: experiment design compliance?" -> "Implementer fixes spec gaps" [label="no"];
    "Implementer fixes spec gaps" -> "Dispatch ML spec reviewer" [label="re-review"];
    "Spec reviewer: experiment design compliance?" -> "Dispatch ML quality reviewer" [label="yes"];
    "Dispatch ML quality reviewer" -> "Quality reviewer: code quality?";
    "Quality reviewer: code quality?" -> "Implementer fixes quality issues" [label="no"];
    "Implementer fixes quality issues" -> "Dispatch ML quality reviewer" [label="re-review"];
    "Quality reviewer: code quality?" -> "L0: VP Static Checks" [label="yes"];
    "L0: VP Static Checks" -> "L0 passed?";
    "L0 passed?" -> "L1: ML Runtime Validator" [label="yes"];
    "L0 passed?" -> "Implementer fixes L0 issues" [label="no"];
    "Implementer fixes L0 issues" -> "L0: VP Static Checks" [label="re-run L0\n(fix>50 lines: rollback)"];
    "L1: ML Runtime Validator" -> "L1 passed?";
    "L1 passed?" -> "L2: ML E2E Validator" [label="yes"];
    "L1 passed?" -> "Implementer fixes L1 issues" [label="no"];
    "Implementer fixes L1 issues" -> "L1: ML Runtime Validator" [label="re-run L1\n(fix>50 lines: rollback)"];
    "L2: ML E2E Validator" -> "L2 passed?";
    "L2 passed?" -> "Record conclusion" [label="yes"];
    "L2 passed?" -> "Implementer fixes L2 issues" [label="no"];
    "Implementer fixes L2 issues" -> "L2: ML E2E Validator" [label="re-run L2\n(fix>50 lines: rollback)"];
    "Record conclusion" -> "More subtasks?";
    "More subtasks?" -> "Dispatch ML implementer subagent" [label="yes"];
    "More subtasks?" -> "Needs long-running training?" [label="no"];
    "Needs long-running training?" -> "Invoke spml:training-handoff" [label="yes\n(hours/days)"];
    "Needs long-running training?" -> "Invoke spml:verification" [label="no\n(already complete)"];
}
```

## ML Implementer Subagent Prompt

```
You are implementing Subtask N: [subtask name]

## Experiment Context

**Overall hypothesis:** [from plan header]
**This subtask's hypothesis:** [specific to this subtask]
**Validation scope:** [which VP levels are enabled]

## Task Description

[FULL TEXT of subtask from plan]

## Code Separation Rule

Core code (model, training, data) must NEVER import from test/validation code
or toolkit. Validation scripts observe core code externally.

## Your Job

1. **Write unit tests** for any custom functions (deterministic code only)
2. **Run unit tests** — verify they fail (TDD red)
3. **Implement core code** (no test/validation imports)
4. **Run unit tests** — verify they pass (TDD green)
5. **Self-review** — check your own code before submission
6. **Commit** with message: "experiment: [subtask description]"

Note: Validation Pyramid (L0/L1/L2) is run by the orchestrator AFTER your code passes reviews. You do NOT run VP yourself.

If this subtask includes evaluation work:
- build one evaluator core shared by checkpoint-based and in-memory entry modes
- keep cadence decisions in trainer code and evaluation execution logic in evaluator code
- do not implement final-epoch-only evaluation as the default for long-running training
- expose mode-aware start/end reporting and boundary errors

## Report Format

- What you implemented
- Unit test results
- Files changed
- Any concerns or questions
```

## ML Spec Reviewer Prompt

```
You are reviewing whether a subtask implementation matches its experiment design.

## Experiment Design

**Hypothesis:** [from plan]
**Independent variable:** [what should change]
**Dependent variable:** [what to measure]
**Control variable:** [what must stay the same]

## Subtask Spec

[FULL TEXT of subtask requirements]

## Your Job

Read the actual code and verify:

**Experiment design compliance:**
- Does the implementation match the stated hypothesis?
- Is ONLY the independent variable changed? (no confounds)
- Are control variables truly unchanged?
- Is the dependent variable being measured correctly?

**Spec compliance (same as standard review):**
- Missing requirements?
- Extra/unneeded work?
- Misunderstandings?

**ML-specific checks:**
- Core code imports from test/validation code? (VIOLATION)
- Validation scripts observe externally? (hooks/wrappers, not modifying core)
- Correct loss function for the task?
- Data preprocessing matches training and evaluation?
- If evaluation is in scope, does the plan/code preserve the split:
  - trainer decides when evaluation runs
  - evaluator decides how evaluation runs
- If evaluation is in scope, are both entry modes present through one shared evaluator core?
- If evaluation is in scope, is evaluation still observable during long runs?

Report:
- ✅ Spec compliant
- ❌ Issues found: [list with file:line references]
```

## ML Quality Reviewer Prompt

```
You are reviewing implementation quality for a completed ML subtask.

Note: Validation Pyramid (L0/L1/L2) runs AFTER this quality review. You review code quality only — VP metrics are the orchestrator's responsibility.

## Your Job

**Code quality (same as standard review):**
- Clean, maintainable code?
- Proper error handling at system boundaries?
- No security issues?

**ML-specific quality:**
- Fixed random seeds where needed?
- Proper CUDA synchronization for timing?
- No data leakage between train/eval?
- Gradient computation correct (detach where needed)?
- If evaluation is in scope, are mode-aware boundary errors and progress signals implemented where they belong?

Report:
- ✅ Approved
- ❌ Issues: [list with severity and file:line references]
```

## Conclusion Recording

After each subtask completes all reviews:

```markdown
### Subtask N Conclusion

**Hypothesis:** [restated]
**Result:** effective / ineffective / inconclusive
**Evidence:**
- [metric]: [actual value] (expected: [threshold])
- [metric]: [actual value] (expected: [threshold])
**Anomalies:** [any unexpected observations]
**Recommendation:** [proceed / investigate further / abandon direction]
```

Record this in the plan document or a separate experiment log.

## Training Handoff Decision

After all subtasks complete, before invoking verification, check:

**Does this experiment need a long-running execution phase?** (training for hours/days, large-scale data processing, full evaluation sweep)

- **Yes** → Invoke `spml:training-handoff`. This generates:
  - Production training script with human-readable logging (including MFU, gradient norms, etc.)
  - `experiment-context.md` with VP baseline metrics
  - `watchdog-prompt.md` for monitoring in a separate session
  - Verification happens LATER, after training completes (via `spml:training-resume`)

- **No** → Invoke `spml:verification` directly. The experiment is already complete within this session.

**How to decide:** If VP validation (L1/L2) ran a shortened version of training (e.g., 1250 steps instead of 100K), and the experiment goal requires full-scale results, then a long-running phase is needed.

When the long-running phase includes evaluation, downstream checks should confirm:
- in-training evaluation fires at the planned step cadence
- checkpoint-based evaluation reports checkpoint load behavior
- in-training evaluation reports that it is using in-memory state
- evaluation start/end messages and progress output appear as runtime checks, not optional niceties
- evaluation errors surface with mode-aware context at the evaluation boundary

## Red Flags

**Never:**
- Skip Validation Pyramid execution
- Accept VP "pass" without checking actual numbers
- Let implementer skip unit tests for custom code
- Proceed when VP layer fails (trigger diagnostics instead)
- Change control variables in a subtask (confounds the experiment)
- Record "effective" without VP evidence

**Always:**
- Record actual metric values (not just pass/fail)
- Note anomalies even when passing
- Keep core code free of test/validation imports
- Fixed random seeds for reproducibility

## Integration

- **spml:experiment-planning** — Creates the plan this skill executes
- **spml:validation-pyramid** — Defines the 3-level VP orchestration
- **spml:ml-static-checks** — L0 static analysis (dispatched as subagent after quality review)
- **spml:ml-runtime-validator** — L1 runtime validation (orchestrator invokes after L0)
- **spml:ml-e2e-validator** — L2 E2E pipeline validation (orchestrator invokes after L1)
- **spml:diagnostics** — Called when VP check fails
- **spml:training-handoff** — Called after all subtasks complete IF long-running training is needed
- **spml:verification** — Called after all subtasks complete IF experiment is already done (no long-running phase)
