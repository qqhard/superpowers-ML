---
name: subagent-dev
description: Use when executing ML experiment plans with subagents - adapts subagent-driven-development with 3-level Validation Pyramid (L0 static → L1 runtime → L2 E2E), experiment-aware reviews, and conclusion recording
---

# ML Subagent-Driven Development

Execute ML experiment plans by dispatching fresh subagent per subtask, with ML-adapted review criteria: spec compliance (does it match experiment design?), quality review (did Validation Pyramid pass?), and conclusion recording.

**Core principle:** Fresh subagent per subtask + experiment-aware review + conclusion recording = correct implementations with trustworthy conclusions.

**Adapted from:** subagent-driven-development. Key changes:
- Validation Pyramid runs AFTER code reviews as 3 separate orchestrator-dispatched stages (L0 → L1 → L2)
- L0: ML Code Reviewer subagent checks static ML correctness
- L1: Runtime validation (minutes-level training run with metrics collection)
- L2: E2E pipeline validation (1-5 steps per stage)
- Spec reviewer checks experiment design compliance (hypothesis, variable control)
- Quality reviewer checks code quality (VP results checked by orchestrator, not quality reviewer)
- Each subtask records: metric data, conclusion, anomaly log
- Shared fix loop: fail → Implementer fixes → re-run level → 5 failures → user intervention
- Large fix rollback: fix > 50 lines → re-run all prior reviews + VP levels

## When to Use

- You have an ML experiment plan (from experiment-planning)
- Subtasks are mostly independent
- You want to stay in this session (vs. executing-plans in parallel session)

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
        "L0: ML Code Reviewer" [shape=box style=filled fillcolor=lightyellow];
        "L0 passed?" [shape=diamond];
        "L1: ML Runtime Validator" [shape=box style=filled fillcolor=lightyellow];
        "L1 passed?" [shape=diamond];
        "L2: ML E2E Validator" [shape=box style=filled fillcolor=lightyellow];
        "L2 passed?" [shape=diamond];
        "Implementer fixes VP issues" [shape=box];
        "Record conclusion" [shape=box style=filled fillcolor=lightgreen];
    }

    "Read plan, extract subtasks, create TodoWrite" [shape=box];
    "More subtasks?" [shape=diamond];
    "Invoke verification" [shape=box style=filled fillcolor=lightblue];

    "Read plan, extract subtasks, create TodoWrite" -> "Dispatch ML implementer subagent";
    "Dispatch ML implementer subagent" -> "Implementer: unit tests + implement";
    "Implementer: unit tests + implement" -> "Dispatch ML spec reviewer";
    "Dispatch ML spec reviewer" -> "Spec reviewer: experiment design compliance?";
    "Spec reviewer: experiment design compliance?" -> "Implementer fixes spec gaps" [label="no"];
    "Implementer fixes spec gaps" -> "Dispatch ML spec reviewer" [label="re-review"];
    "Spec reviewer: experiment design compliance?" -> "Dispatch ML quality reviewer" [label="yes"];
    "Dispatch ML quality reviewer" -> "Quality reviewer: code quality?";
    "Quality reviewer: code quality?" -> "Implementer fixes quality issues" [label="no"];
    "Implementer fixes quality issues" -> "Dispatch ML quality reviewer" [label="re-review"];
    "Quality reviewer: code quality?" -> "L0: ML Code Reviewer" [label="yes"];
    "L0: ML Code Reviewer" -> "L0 passed?";
    "L0 passed?" -> "L1: ML Runtime Validator" [label="yes"];
    "L0 passed?" -> "Implementer fixes VP issues" [label="no"];
    "L1: ML Runtime Validator" -> "L1 passed?";
    "L1 passed?" -> "L2: ML E2E Validator" [label="yes"];
    "L1 passed?" -> "Implementer fixes VP issues" [label="no"];
    "L2: ML E2E Validator" -> "L2 passed?";
    "L2 passed?" -> "Record conclusion" [label="yes"];
    "L2 passed?" -> "Implementer fixes VP issues" [label="no"];
    "Implementer fixes VP issues" -> "L0: ML Code Reviewer" [label="re-run failed level\n(fix > 50 lines:\nrollback to spec review)"];
    "Record conclusion" -> "More subtasks?";
    "More subtasks?" -> "Dispatch ML implementer subagent" [label="yes"];
    "More subtasks?" -> "Invoke verification" [label="no"];
}
```

## ML Implementer Subagent Prompt

```
You are implementing Subtask N: [subtask name]

## Experiment Context

**Overall hypothesis:** [from plan header]
**This subtask's hypothesis:** [specific to this subtask]
**Validation scope:** [which VP layers are enabled]

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

Report:
- ✅ Spec compliant
- ❌ Issues found: [list with file:line references]
```

## ML Quality Reviewer Prompt

```
You are reviewing implementation quality for a completed ML subtask.

## Validation Pyramid Results

[Paste actual results from implementer report]

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
- **spml:ml-code-reviewer** — L0 static analysis (dispatched as subagent after quality review)
- **spml:ml-runtime-validator** — L1 runtime validation (orchestrator invokes after L0)
- **spml:ml-e2e-validator** — L2 E2E pipeline validation (orchestrator invokes after L1)
- **spml:diagnostics** — Called when VP check fails
- **spml:verification** — Called after all subtasks complete
