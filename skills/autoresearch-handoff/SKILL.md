---
name: autoresearch-handoff
description: Use after VP passes when the experiment needs automated iteration — verifies base code, generates research protocol, and produces startup prompt for autonomous exploration
---

# Autoresearch Handoff

## Overview

Bridge between VP validation and automated research iteration. Generates `autoresearch-protocol.md` + `experiences.md` for base code that has already been built and validated.

**Core principle:** Do not rewrite VP-validated code. Handoff verifies readiness, extracts the protocol, and sets up the iteration loop.

<HARD-GATE>
Do NOT hand off without:
1. VP L1 Runtime Validation passed — mandatory for autoresearch. If skipped, STOP and run it now.
2. Design doc contains "## Autoresearch Protocol" section
</HARD-GATE>

## Checklist

1. **Verify VP L1** — record baseline metric
2. **Verify pressure condition termination** — time_limit / epoch_limit logic works
3. **Generate autoresearch-protocol.md** — simplified format
4. **Initialize experiences.md** — table with baseline row
5. **Verify git state** — base code committed
6. **Present launch instructions**

## Step 1: Verify VP L1

Confirm VP L1 passed. Record the evaluation metric as the protocol's baseline value.

## Step 2: Verify Pressure Conditions

Check that time_limit and epoch_limit termination logic exists and works (VP L1 already ran under these conditions).

## Step 3: Generate autoresearch-protocol.md

Extract from design doc's `## Autoresearch Protocol` section. Write to `<experiment-dir>/autoresearch-protocol.md`:

```markdown
# Autoresearch Protocol: <title>

research_question: <from design doc>
max_rounds: <from design doc>
target: <from design doc, or "none">
baseline: <metric> = <value from VP L1>

## Fixed（不可变：代码 + 条件）
- files: <fixed files from design doc>
- time_limit: <value>
- epoch_limit: <value>

## Variable（可变：代码 + 条件）
- files: <variable files from design doc>
- 可调范围: <from design doc>

## Eval
- metric: <name>
- direction: <maximize / minimize>
- command: <eval_command from design doc>
```

## Step 4: Initialize experiences.md

Write to `<experiment-dir>/experiences.md`:

```markdown
# Experiences

best: {metric} = {baseline_value} (R0)
rounds: 0 / {max_rounds}
status: not_started

| Round | Strategy | Compliance | Result | Verdict | Insight | Note |
|-------|----------|------------|--------|---------|---------|------|
| 0 | baseline | ✅ | {baseline_value} | — | initial | {user hints if any} |
```

## Step 5: Verify Git State

```bash
git log --oneline -1  # base code committed
```

## Step 6: Present Launch Instructions

```
Handoff complete:
- Protocol: <experiment-dir>/autoresearch-protocol.md
- Experiences: <experiment-dir>/experiences.md
- Baseline: {metric} = {value}
- Max rounds: {N}

To start — open a new session and say:

  run autoresearch at <experiment-dir>
```
