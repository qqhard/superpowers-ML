---
name: using-spml
description: Use when starting any conversation - establishes how to find and use ML skills, requiring host-appropriate skill loading before ANY response including clarifying questions
---

<EXTREMELY-IMPORTANT>
If you think there is even a 1% chance a skill might apply to what you are doing, you ABSOLUTELY MUST invoke the skill.

IF A SKILL APPLIES TO YOUR TASK, YOU DO NOT HAVE A CHOICE. YOU MUST USE IT.

This is not negotiable. This is not optional. You cannot rationalize your way out of this.
</EXTREMELY-IMPORTANT>

## How to Access Skills

Use the mechanism that matches your host:

- In Claude Code: use the `Skill` tool to load skills, `TodoWrite` to track
  checklist items, and `Task` for subagents when a skill asks for them.
- In Codex: rely on native skill discovery from the configured skills
  directory, use `update_plan` to track checklist items, and use native
  subagents.
- In any host: once a skill is loaded, follow it directly and do not treat
  `SKILL.md` like an ordinary project file.

# Using Skills

## ML Experiment Gate (CHECK FIRST)

**Before considering any `spml:*` skill, determine whether the current task is conducting an ML experiment.** SPML skills are designed for running experiments, not for building software — even software that happens to involve ML code.

**IS an ML experiment** (→ proceed with SPML skills):
- Actually training or fine-tuning a model to observe results
- Running hyperparameter sweeps or ablation studies
- Evaluating a trained model's performance on a dataset
- Debugging a live training run (convergence failures, NaN losses, gradient issues)
- Preparing a dataset **for an imminent experiment** (not building a reusable data pipeline)

**Is NOT an ML experiment** (→ skip ALL `spml:*` skills, defer to `superpowers:*` equivalents):
- Building or refactoring ML frameworks, toolkits, or scaffolding
- Implementing model architectures, loss functions, or data loaders as software components
- Writing tests, CI/CD, or infrastructure for ML code
- Developing training pipelines, evaluation harnesses, or experiment tracking systems
- Any task where the goal is shipping code, not observing experimental outcomes

**The key question: Is the goal to observe an experimental outcome, or to ship working software?**
- Observe outcome → SPML
- Ship software → Superpowers (even if the software is ML-related)

**Experiment directory override:** If an upstream brainstorm has already established an experiment directory, ALL downstream artifacts (plans, docs, design docs) MUST be saved under that experiment directory — even when deferring to `superpowers:*` skills like `superpowers:writing-plans`. Never fall back to the default `docs/plans/` path when an experiment directory has been set.

## Experiment Directory Detection

Before routing to any skill, check if the user's request references an existing experiment directory.

**Detection:**
1. **Explicit path** — user mentions a directory like "experiments/xxx" or "modify the gumbel experiment"
2. **Implicit** — user references an experiment and there's only one experiment directory

**If an experiment directory is identified, scan it:**

<HARD-GATE>
When an experiment directory with existing artifacts is detected, you MUST read all existing design docs and plans BEFORE invoking any skill. Pass their content as context to the invoked skill.

Do NOT create new design/plan files when existing ones can be revised.
Do NOT let any skill skip reading existing artifacts.
</HARD-GATE>

**State detection:**

```
Scan <experiment-dir>/plans/:
  - *-design.md exists? → has_design = true
  - non-design *.md exists? → has_plan = true

Scan <experiment-dir>/:
  - *.py exists? → has_code = true
```

**Routing based on state:**

| has_design | has_plan | has_code | Action |
|------------|----------|----------|--------|
| false | false | false | New experiment: invoke `spml:ml-brainstorming` (new mode) |
| true | false | false | Read design → invoke `spml:experiment-planning` |
| true | true | false | Read plan → invoke `spml:ml-subagent-dev` |
| true | true | true | Read design → invoke `spml:ml-brainstorming` (revision mode) |

**Last row is critical:** When a full experiment exists and the user wants changes, ALWAYS start from the design. Changes propagate top-down: design → plan → code. Direct code changes without design/plan updates are forbidden.

## The Rule

**Load relevant or requested skills BEFORE any response or action.** Even a 1%
chance a skill might apply means that you should load it to check using the
host-appropriate mechanism. If a loaded skill turns out to be wrong for the
situation, you don't need to use it.

```dot
digraph skill_flow {
    "User message received" [shape=doublecircle];
    "About to EnterPlanMode?" [shape=doublecircle];
    "Already brainstormed?" [shape=diamond];
    "Invoke spml:ml-brainstorming" [shape=box];
    "Is this an ML experiment?" [shape=diamond, style=bold, color=red];
    "Skip all spml:* skills\nDefer to superpowers:*" [shape=box, style=dashed];
    "Might any spml skill apply?" [shape=diamond];
    "Load relevant skill\n(Skill tool / native discovery)" [shape=box];
    "Announce: 'Using [skill] to [purpose]'" [shape=box];
    "Has checklist?" [shape=diamond];
    "Create checklist tracking item\n(TodoWrite / update_plan)" [shape=box];
    "Follow skill exactly" [shape=box];
    "Respond (including clarifications)" [shape=doublecircle];

    "About to EnterPlanMode?" -> "Already brainstormed?";
    "Already brainstormed?" -> "Invoke spml:ml-brainstorming" [label="no"];
    "Already brainstormed?" -> "Is this an ML experiment?" [label="yes"];
    "Invoke spml:ml-brainstorming" -> "Is this an ML experiment?";

    "User message received" -> "Is this an ML experiment?";
    "Is this an ML experiment?" -> "Skip all spml:* skills\nDefer to superpowers:*" [label="no"];
    "Skip all spml:* skills\nDefer to superpowers:*" -> "Respond (including clarifications)";
    "Is this an ML experiment?" -> "Might any spml skill apply?" [label="yes"];
    "Might any spml skill apply?" -> "Load relevant skill\n(Skill tool / native discovery)" [label="yes, even 1%"];
    "Might any spml skill apply?" -> "Respond (including clarifications)" [label="definitely not"];
    "Load relevant skill\n(Skill tool / native discovery)" -> "Announce: 'Using [skill] to [purpose]'";
    "Announce: 'Using [skill] to [purpose]'" -> "Has checklist?";
    "Has checklist?" -> "Create checklist tracking item\n(TodoWrite / update_plan)" [label="yes"];
    "Has checklist?" -> "Follow skill exactly" [label="no"];
    "Create checklist tracking item\n(TodoWrite / update_plan)" -> "Follow skill exactly";
}
```

## Red Flags

These thoughts mean STOP—you're rationalizing:

| Thought | Reality |
|---------|---------|
| "This is just a simple question" | Questions are tasks. Check for skills. |
| "I need more context first" | Skill check comes BEFORE clarifying questions. |
| "Let me explore the codebase first" | Skills tell you HOW to explore. Check first. |
| "I can check git/files quickly" | Files lack conversation context. Check for skills. |
| "Let me gather information first" | Skills tell you HOW to gather information. |
| "This doesn't need a formal skill" | If a skill exists, use it. |
| "I remember this skill" | Skills evolve. Read current version. |
| "This doesn't count as a task" | Action = task. Check for skills. |
| "The skill is overkill" | Simple things become complex. Use it. |
| "I'll just do this one thing first" | Check BEFORE doing anything. |
| "This feels productive" | Undisciplined action wastes time. Skills prevent this. |
| "I know what that means" | Knowing the concept ≠ using the skill. Invoke it. |
| "ML is different, I can skip the process" | ML needs MORE process, not less. Follow it. |

## Skill Priority

When multiple skills could apply, use this order:

1. **Process skills first** (`spml:ml-brainstorming`, `spml:diagnostics`) - these determine HOW to approach the task
2. **Validation skills second** (`spml:validation-pyramid`, `spml:ml-static-checks`, `spml:ml-runtime-validator`, `spml:ml-e2e-validator`) - these verify correctness
3. **Implementation skills third** (`spml:experiment-planning`, `spml:ml-subagent-dev`) - these guide execution

"Let's train X" -> `spml:ml-brainstorming` first, then `spml:experiment-planning`.
"Training isn't converging" -> `spml:diagnostics` first, then validation skills.
"MFU is too low" -> `spml:diagnostics` first, then `spml:ml-runtime-validator`.

## Skill Types

**Rigid** (validation-pyramid, diagnostics): Follow exactly. Don't adapt away discipline.

**Flexible** (framework knowledge): Adapt principles to context.

The skill itself tells you which.

## Core Principle for ML

**In ML, code running without errors does NOT mean it's correct.** "Not working" is reasonable, but the process must be correct. Always validate through the Validation Pyramid before concluding an experiment.

When validation or evaluation is part of the task, plan it explicitly upstream. Treat evaluation as a first-class subtask with:
- step-based cadence by default
- `full validation` scope by default unless explicitly overridden
- both checkpoint-based and in-memory entry modes
- one shared evaluator core
- visible progress and mode-aware error reporting during long-running evaluation

## User Instructions

Instructions say WHAT, not HOW. "Train X" or "Fix convergence" doesn't mean skip workflows.
