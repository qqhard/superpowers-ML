---
name: autoresearch-handoff
description: Use after VP passes when the experiment needs automated iteration — verifies base code, generates research protocol, and produces startup prompt for autonomous exploration
---

# Autoresearch Handoff

## Overview

Bridge between VP validation and automated research iteration. Generates research artifacts (autoresearch-protocol.md + autoresearch-prompt.md + experiences.md) for base code that has already been built and validated by the upstream flow.

**Core principle:** Do not rewrite VP-validated code. The base code was built by ml-subagent-dev, tested by VP L0/L1. Handoff's job is to verify it's research-ready, extract the protocol, and set up the iteration loop — not to modify code.

<HARD-GATE>
Do NOT hand off without:
1. All enabled VP layers passed
2. Base code runs under pressure conditions (VP L1 verified this)
3. Pressure condition termination code exists and works (part of Fixed Conditions)
4. Design doc contains "## Autoresearch Protocol" section
</HARD-GATE>

## When to Use

- All VP checks passed for the experiment
- Design doc contains `## Autoresearch Protocol` section
- User chose "Research" at ml-subagent-dev Post-Completion Gate

## Checklist

1. **Verify VP L1 completion** — all enabled layers passed, record baseline metric
2. **Verify pressure condition termination code** — base code has working time_limit / epoch_limit logic
3. **Generate autoresearch-protocol.md** — extract 6 elements from design doc
4. **Initialize experiences.md** — with Summary header and baseline record
5. **Generate autoresearch-prompt.md** — startup prompt for new session
6. **Verify git state** — base code committed as initial checkpoint
7. **Present launch instructions** — show user how to start

## Step 1: Verify VP L1 Completion

Confirm all VP layers enabled in the brainstorm design doc have passed. Record the evaluation metric result as the protocol's baseline value.

```
VP Status:
  L0 Static:  ✅ All mandatory checks passed
  L1 Runtime:  ✅ [metrics] — pipeline complete, evaluation produced baseline
  Baseline metric: {metric} = {value}
```

## Step 2: Verify Pressure Condition Termination Code

The base code must contain working termination logic for the pressure conditions defined in the Autoresearch Protocol section of the design doc. This code is part of Fixed Conditions — Agent A must not modify it during iterations.

Check that:
- [ ] Time limit argument exists and terminates training when reached
- [ ] Epoch limit argument exists and terminates training when reached
- [ ] Whichever triggers first takes effect
- [ ] VP L1 run completed under these conditions (it did, since VP L1 ran the full pipeline)

## Step 3: Extract Protocol from Design Doc

Read the design doc's `## Autoresearch Protocol` section. Extract all 6 elements:
- Fixed Conditions
- Pressure Conditions (time_limit, epoch_limit)
- Variable Conditions
- Evaluation (metric, direction, eval_command)
- Termination (max_rounds, target)
- Agent Boundary (agent_a tasks, agent_b tasks)

## Step 4: Generate autoresearch-protocol.md

Write to `<experiment-dir>/autoresearch-protocol.md` using the protocol template, filled with values from Step 3:

```markdown
# Autoresearch Protocol: <title from design doc>

## Research Question
<from design doc's experiment design section>

## Environment
- Base code: <experiment-dir path>
- Dataset: <from design doc>
- Framework: <from design doc>

## Fixed Conditions
<from design doc's Autoresearch Protocol section>

## Pressure Conditions
- time_limit: <value>
- epoch_limit: <value>
- Whichever triggers first

## Variable Conditions
<from design doc's Autoresearch Protocol section>

## Evaluation
- metric: <name>
- direction: <maximize / minimize>
- eval_command: "<command>"
- baseline: <value from VP L1 run>

## Termination
- max_rounds: <N>
- target: <value or "none">

## Agent Boundary
- agent_a: <list>
- agent_b: <list>
```

## Step 5: Initialize experiences.md

Write to `<experiment-dir>/experiences.md`:

```markdown
# Experiment Experiences

## Summary
- Best result: {metric} = {baseline_value} (Baseline)
- Total rounds: 0 / {max_rounds}
- Status: not_started
```

## Step 6: Generate autoresearch-prompt.md

Write to `<experiment-dir>/autoresearch-prompt.md`:

```markdown
I need you to run an automated research loop, iterating on ML code to optimize a target metric.

## Setup
1. Read `<experiment-dir>/autoresearch-protocol.md` for the full research protocol
2. Read `<experiment-dir>/experiences.md` for the current state
3. Verify git state: base code commit exists

## Your Behavior
Use the spml:autoresearch skill. It will guide you through:
- Reading the protocol and verifying the worktree
- Dispatching Agent A (designer/coder) and Agent B (evaluator/reviewer) each round
- Managing git state: commit on improvement, rollback on failure
- Tracking all experience in experiences.md
- Terminating when target is reached or max rounds exhausted
- Producing a final report
```

## Step 7: Verify Git State and Present Launch Instructions

Confirm the base code is committed:
```bash
git log --oneline -1  # should show a recent commit with base code
```

Then show the artifact summary and **print the full autoresearch-prompt.md content** in the conversation so the user can copy-paste:

```
Handoff complete. All artifacts generated:
- Base code: <experiment-dir>/ (VP L1 validated)
- Protocol: <experiment-dir>/autoresearch-protocol.md
- Experience log: <experiment-dir>/experiences.md
- Startup prompt: <experiment-dir>/autoresearch-prompt.md
- Baseline: {metric} = {value}
- Max rounds: {N}, Target: {target or "none"}

To start — copy the prompt below into a new agent session:
```

Then output the full content of autoresearch-prompt.md in a fenced code block.

## Integration

- **spml:ml-subagent-dev** — Triggers handoff when user chooses "Research" at Post-Completion Gate
- **spml:autoresearch** — The startup prompt references this skill
- **spml:ml-brainstorming** — Design doc's Autoresearch Protocol section is the input
