# Autoresearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add autoresearch capabilities to SPML — a protocol-driven automated experiment iteration loop with two subagents (designer + evaluator), git-managed code state, and experience accumulation.

**Architecture:** Three skill files (autoresearch-handoff, autoresearch, ml-brainstorming modification) plus a Post-Completion Gate modification in ml-subagent-dev. An integration test with pseudo-data validates the full loop end-to-end.

**Tech Stack:** SPML skill markdown, bash test scripts, Python (numpy only for test base code)

---

## File Structure

```
skills/
├── autoresearch-handoff/
│   └── SKILL.md                    # NEW — handoff skill
├── autoresearch/
│   └── SKILL.md                    # NEW — supervisor skill
├── ml-brainstorming/
│   └── SKILL.md                    # MODIFY — add autoresearch detection
├── ml-subagent-dev/
│   └── SKILL.md                    # MODIFY — add Research option to Post-Completion Gate

tests/
├── autoresearch/
│   ├── run-test.sh                 # NEW — integration test runner
│   ├── verify.sh                   # NEW — post-run verification
│   └── base-project/               # NEW — template base code
│       ├── train.py
│       ├── evaluate.py
│       └── autoresearch-protocol.md
```

---

### Task 1: Create `autoresearch-handoff` Skill

**Files:**
- Create: `skills/autoresearch-handoff/SKILL.md`

This skill bridges VP validation and autoresearch execution. It verifies base code readiness, generates the protocol document, initializes the experience log, and produces the startup prompt for a new session.

- [ ] **Step 1: Create the skill file with frontmatter and overview**

```markdown
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
```

- [ ] **Step 2: Add the checklist and Steps 1-3 (verification steps)**

```markdown
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
```

- [ ] **Step 3: Add Steps 4-7 (generation steps) with protocol template**

```markdown
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
```

- [ ] **Step 4: Assemble the complete SKILL.md file**

Combine all sections from Steps 1-3 into the final `skills/autoresearch-handoff/SKILL.md`.

- [ ] **Step 5: Commit**

```bash
git add skills/autoresearch-handoff/SKILL.md
git commit -m "feat: add autoresearch-handoff skill"
```

---

### Task 2: Create `autoresearch` Supervisor Skill

**Files:**
- Create: `skills/autoresearch/SKILL.md`

This is the core skill — the supervisor that runs the outer loop in a new session. It dispatches Agent A and Agent B as subagents, manages git state, and accumulates experience.

- [ ] **Step 1: Create the skill file with frontmatter, overview, and hard gates**

```markdown
---
name: autoresearch
description: Use when running an automated research loop — reads protocol, dispatches designer/evaluator subagents, manages git state, accumulates experience across iterations
---

# Autoresearch

## Overview

Automated research supervisor. Reads a research protocol, then iterates: dispatch Agent A (design + code + train) → dispatch Agent B (evaluate + review + record) → commit or rollback based on verdict → repeat until termination.

**Core principle:** Keep the loop running. The Supervisor manages form and process — it does NOT design strategies or write code. Agent A and Agent B do the creative work; the Supervisor controls git, checks termination, and recovers from failures.

<HARD-GATE>
## Git Control

You MUST be the ONLY entity that performs git write operations (commit, checkout, reset).
Agent A and Agent B have file read/write and bash permissions but NO git write permissions.
All git state changes go through the Supervisor.
</HARD-GATE>

<HARD-GATE>
## Monitoring Loop Mechanism

You MUST use subagents (Agent tool) as the ONLY way to dispatch Agent A and Agent B.
Each agent is a fresh subagent per round — no shared context between rounds.
Experience transfer happens through files (experiences.md, git history), not agent memory.
</HARD-GATE>
```

- [ ] **Step 2: Add startup procedure**

```markdown
## When to Use

- User has pasted an autoresearch prompt from autoresearch-handoff
- An autoresearch-protocol.md exists in the experiment directory

## Startup

1. **Read `autoresearch-protocol.md`** — understand research question, fixed/variable/pressure conditions, evaluation method, termination criteria, agent boundary
2. **Verify worktree state:**
   - Base code commit exists (`git log --oneline -1`)
   - `experiences.md` exists and has Summary section
3. **Check for resume:**
   - Read `experiences.md` Summary → extract `Total rounds` and `Status`
   - If Status is `running` and Total rounds > 0: resuming from interruption
     - Current round = Total rounds + 1
     - Verify git HEAD matches latest committed improvement (or baseline if no improvements yet)
   - If Status is `not_started`: fresh start, round = 1
4. **Announce:** "Starting autoresearch: {research_question}. Round {current} / {max_rounds}. Baseline: {metric} = {baseline}."
5. **Enter main loop**
```

- [ ] **Step 3: Add main loop**

```markdown
## Main Loop

```
for round in current_round..max_rounds:

  1. DISPATCH AGENT A
  2. DISPATCH AGENT B
  3. ACT ON VERDICT
  4. CHECK TERMINATION
  5. REPORT PROGRESS
```

### Step 1: Dispatch Agent A

Dispatch a fresh subagent with the following prompt structure. Agent A has full file read/write and bash access but NO git write access.

```
You are an ML researcher. Your task is to improve {metric} ({direction}).
This is Round {round} of {max_rounds}.

## File Paths
- Protocol: {protocol_path}  — read this for research question, constraints, and your degrees of freedom
- Experience log: {experiences_path}  — read this for all past strategies and results
- Codebase: {worktree_path}

## Instructions
- Read the protocol first to understand what you can and cannot change
- Read experiences.md to learn from past rounds — decide yourself how much to read
- Git history contains successful strategies as commits — use git log / git diff to study what code changes worked
- Do NOT modify termination logic or evaluation logic (these are Fixed Conditions in the protocol)
- Write your strategy description to strategy.md BEFORE modifying any code
- After modifying code, run training: the pressure conditions will auto-terminate at the limit
- When training completes, report "Training complete" as your final message
```

Wait for Agent A to complete. If Agent A times out or crashes, see Anomaly Recovery.

### Step 2: Dispatch Agent B

Dispatch a fresh subagent with the following prompt structure. Agent B has full file read/write and bash access but NO git write access.

```
You are an ML experiment evaluator. Your task is to evaluate Round {round}'s result and record experience.

## File Paths
- Protocol: {protocol_path}  — read this for evaluation method and metric details
- Experience log: {experiences_path}  — read this for current best result in Summary section
- Strategy: {worktree_path}/strategy.md  — read this for what Agent A tried this round

## Instructions
1. Read the protocol's Evaluation section for the eval_command and metric details
2. Run the evaluation command from the protocol
3. Read the current best result from experiences.md Summary section
4. Compare this round's result against the best result
5. Append a new Round entry to experiences.md:
   - **Strategy**: (from strategy.md)
   - **Result**: {metric} = {value}
   - **Verdict**: ✅ committed or ❌ rolled back
   - **Insight**: what worked or didn't and why
6. Your final message MUST be exactly one of these two formats:
   VERDICT: improved {metric}={value}
   VERDICT: not_improved {metric}={value}
```

Wait for Agent B to complete. Parse the VERDICT line from Agent B's response.

### Step 3: Act on Verdict

**If improved:**
```bash
# Save experiences.md first (Agent B already updated it)
cp experiences.md /tmp/experiences_backup.md

# Commit all changes including updated experiences.md
git add -A
git commit -m "autoresearch: round {round} — {metric}={value} (improved)"

# Update Summary section
# (edit experiences.md: update Best result line, increment Total rounds)
```

**If not_improved:**
```bash
# Save experiences.md (Agent B already appended the failed round entry)
cp experiences.md /tmp/experiences_backup.md

# Rollback all code changes
git checkout -- .
git clean -fd

# Restore experiences.md with the new round entry
cp /tmp/experiences_backup.md experiences.md

# Update Summary section
# (edit experiences.md: increment Total rounds only, keep Status as running)
```

### Step 4: Check Termination

- Parse the metric value from Agent B's VERDICT line
- If `direction == minimize` and `metric_value <= target`: terminate with reason `target_reached`
- If `direction == maximize` and `metric_value >= target`: terminate with reason `target_reached`
- If `round == max_rounds`: terminate with reason `completed`
- Otherwise: continue to next round

### Step 5: Report Progress

After each round, output a status line:

```
Round {round}/{max_rounds}: {metric}={value} — {verdict}
  Best so far: {metric}={best_value} (Round {best_round})
  Trend: {improving / plateauing / oscillating}
```

Trend detection:
- **improving**: at least one of the last 3 rounds was "improved"
- **plateauing**: last 3+ rounds all "not_improved"
- **oscillating**: alternating improved/not_improved in last 4 rounds
```

- [ ] **Step 4: Add anomaly recovery**

```markdown
## Anomaly Recovery

### Agent Timeout / Crash

If Agent A or Agent B fails (subagent returns error or times out):

1. Record in experiences.md:
   ```
   ## Round {N}
   - **Strategy**: (agent_error — Agent {A/B} failed)
   - **Result**: N/A
   - **Verdict**: ⚠️ skipped (agent error)
   - **Insight**: {error message if available}
   ```
2. Rollback any partial code changes: `git checkout -- . && git clean -fd`
3. Restore experiences.md (same backup/restore pattern as not_improved)
4. Retry the round ONCE. If retry also fails, skip this round and continue to next.

### Session Interruption Recovery

On startup, if `experiences.md` Summary shows `Status: running` with `Total rounds > 0`:
1. Read the last Round entry to understand where we left off
2. Verify git HEAD is consistent (matches latest committed improvement)
3. If the last round entry has no Verdict: the interruption happened mid-round. Treat as failed round, rollback any uncommitted changes, continue from that round number.
4. If the last round has a Verdict: continue from next round.

### Consecutive Failures

If 5 consecutive rounds are "not_improved":
- Output a plateau warning: "⚠️ Plateau detected: 5 consecutive rounds without improvement. Continuing, but consider reviewing the protocol's Variable Conditions for wider exploration."
- Continue the loop (do not stop).
```

- [ ] **Step 5: Add final report and integration**

```markdown
## Final Report

When the loop terminates (target reached or max rounds), update experiences.md Summary:
- Set Status to `target_reached` or `completed`

Then output the final report:

```
# Autoresearch Complete

## Result
- Best: {metric} = {value} (Round {N})
- Baseline: {metric} = {baseline_value}
- Improvement: {delta} ({percentage}%)
- Rounds: {completed} / {max_rounds}
- Termination reason: {target_reached / completed}

## Key Insights
<Read experiences.md and distill top 3-5 insights across all rounds>

## Current State
- git HEAD = best performing code
- All experience recorded in experiences.md
- Worktree: {path}
```

## Integration

- **spml:autoresearch-handoff** — Generates the protocol and startup prompt that triggers this skill
- **spml:ml-brainstorming** — Design doc's Autoresearch Protocol section defines the protocol content
- **spml:ml-subagent-dev** — Post-Completion Gate routes to autoresearch-handoff
```

- [ ] **Step 6: Assemble the complete SKILL.md file**

Combine all sections from Steps 1-5 into the final `skills/autoresearch/SKILL.md`.

- [ ] **Step 7: Commit**

```bash
git add skills/autoresearch/SKILL.md
git commit -m "feat: add autoresearch supervisor skill"
```

---

### Task 3: Modify `ml-brainstorming` — Add Autoresearch Detection

**Files:**
- Modify: `skills/ml-brainstorming/SKILL.md`

Add autoresearch scenario detection during the clarifying questions phase, and guide the user to define the 6 protocol elements when detected.

- [ ] **Step 1: Add autoresearch detection section after "Experiment design" subsection (after line 144)**

Insert after the `### Experiment design (when applicable)` section:

```markdown
### Autoresearch detection (when applicable)
If the user's need matches these patterns, suggest autoresearch as an option:
- Goal is to search/optimize rather than validate a single hypothesis
- Multiple iterative attempts expected
- "Find the best X" rather than "test whether X works"

When detected, ask:
> "This sounds like it could benefit from autoresearch — automated iteration where an agent tries strategies, evaluates them, and learns from the results. The agent would iterate autonomously within constraints you define. Want to set this up?"

If the user agrees, ask the following additional questions (one at a time, in order):

1. **Fixed Conditions** — "What must NOT change between iterations?" (model architecture, dataset, specific code modules, etc.)
2. **Pressure Conditions** — "What limits each iteration for fairness? (e.g., 5 minutes per round, 1 epoch per round)"
3. **Variable Conditions** — "What can the agent freely adjust?" (learning rate, optimizer, augmentation, loss function, etc.)
4. **Evaluation** — "What metric determines success? How is it measured?" (metric name, direction, eval command)
5. **Termination** — "When should the loop stop?" (max rounds, target metric value)
6. **Agent Boundary** — "Default: Agent A designs+codes+trains, Agent B evaluates+reviews+records. Want to adjust?" (has default, user can skip)
```

- [ ] **Step 2: Add Autoresearch Protocol section to design doc template**

In the "After the Design" → "Documentation" section, add guidance for when autoresearch is detected:

```markdown
When autoresearch is detected, the design doc includes an additional section:

```markdown
## Autoresearch Protocol

### Fixed Conditions
<from user answers>

### Pressure Conditions
- time_limit: <from user>
- epoch_limit: <from user>

### Variable Conditions
<from user answers>

### Evaluation
- metric: <from user>
- direction: maximize / minimize
- eval_command: <from user or derived from base code>

### Termination
- max_rounds: <from user>
- target: <from user, optional>

### Agent Boundary
- agent_a: ["design", "code", "train"]
- agent_b: ["evaluate", "review", "record"]
```

This section is the routing signal: downstream `ml-subagent-dev` will present the "Research" option at Post-Completion Gate when it detects this section.
```

- [ ] **Step 3: Verify no conflicts with existing content**

Read through the modified SKILL.md to ensure:
- The new sections don't duplicate existing content
- The flow still makes sense (autoresearch detection happens during clarifying questions, protocol goes in design doc)
- Checklist step 3 "Ask clarifying questions" naturally encompasses the autoresearch questions

- [ ] **Step 4: Commit**

```bash
git add skills/ml-brainstorming/SKILL.md
git commit -m "feat(brainstorming): add autoresearch scenario detection and protocol definition"
```

---

### Task 4: Modify `ml-subagent-dev` — Add Research Option to Post-Completion Gate

**Files:**
- Modify: `skills/ml-subagent-dev/SKILL.md`

Add a third "Research" option to the Post-Completion Gate that routes to `autoresearch-handoff`.

- [ ] **Step 1: Modify the Post-Completion Gate section**

Replace the current Post-Completion Gate presentation (lines 369-376) with:

```markdown
First, check if the brainstorm design doc contains a `## Autoresearch Protocol` section.

**If Autoresearch Protocol section exists**, present to the user:

> All subtasks complete. VP passed. Next step:
>
> 1. **Research** — automated experiment iteration. I will invoke spml:autoresearch-handoff to generate the research protocol and startup prompt for autonomous exploration.
> 2. **Train** — needs long-running training (hours/days). I will invoke spml:training-handoff to generate experiment-context.md + watchdog-prompt.md for a new monitoring session.
> 3. **Done** — experiment is already complete within this session. I will invoke spml:verification.
>
> Which one?

**If no Autoresearch Protocol section**, present the original two options:

> All subtasks complete. VP passed. Next step:
>
> 1. **Train** — needs long-running training (hours/days). I will invoke spml:training-handoff to generate experiment-context.md + watchdog-prompt.md for a new monitoring session.
> 2. **Done** — experiment is already complete within this session. I will invoke spml:verification.
>
> Which one?
```

- [ ] **Step 2: Add the Research option handler after the existing handlers**

After the "User chooses Done" handler (line 384), add:

```markdown
- **User chooses Research** → Invoke `spml:autoresearch-handoff`. This generates:
  - `autoresearch-protocol.md` with research constraints and evaluation criteria
  - `autoresearch-prompt.md` for starting the research loop in a new session
  - `experiences.md` initialized with baseline
  - Verification happens LATER, after autoresearch completes (review experiences.md and git HEAD in new session)
```

- [ ] **Step 3: Update the Integration section**

Add to the Integration list at the end of the file:

```markdown
- **spml:autoresearch-handoff** — Called after Post-Completion Gate if user chooses Research
```

- [ ] **Step 4: Commit**

```bash
git add skills/ml-subagent-dev/SKILL.md
git commit -m "feat(subagent-dev): add Research option to Post-Completion Gate for autoresearch"
```

---

### Task 5: Create Integration Test Base Project

**Files:**
- Create: `tests/autoresearch/base-project/train.py`
- Create: `tests/autoresearch/base-project/evaluate.py`
- Create: `tests/autoresearch/base-project/autoresearch-protocol.md`
- Create: `tests/autoresearch/base-project/experiences.md`

Create the pseudo-data base project that the integration test will use. This is a polynomial fitting task — no GPU, numpy only, completes in seconds.

- [ ] **Step 1: Create train.py**

```python
#!/usr/bin/env python3
"""Polynomial fitting trainer for autoresearch integration test.

Fits a polynomial to sin(x) sample points using numpy least squares.
Supports pressure conditions: --epoch-limit and --time-limit.

Fixed Conditions (do NOT modify):
- Dataset: sin(x) on [-pi, pi], 100 train points, seed=42
- Termination logic: epoch limit and time limit
- Output format: result.json with coefficients and train_mse
"""

import argparse
import json
import time
import numpy as np


def generate_data(n_points, seed=42):
    """Generate sin(x) training data. FIXED — do not modify."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-np.pi, np.pi, n_points)
    y = np.sin(x) + rng.normal(0, 0.05, n_points)
    return x, y


def fit_polynomial(x, y, degree=2):
    """Fit polynomial of given degree. This is the Variable Condition."""
    coeffs = np.polyfit(x, y, degree)
    return coeffs


def compute_mse(x, y, coeffs):
    """Compute mean squared error."""
    y_pred = np.polyval(coeffs, x)
    return float(np.mean((y - y_pred) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree (Variable Condition)")
    parser.add_argument("--epoch-limit", type=int, default=1,
                        help="Max epochs (Pressure Condition — FIXED)")
    parser.add_argument("--time-limit", type=float, default=10.0,
                        help="Max seconds (Pressure Condition — FIXED)")
    args = parser.parse_args()

    start_time = time.time()

    # Generate training data (FIXED)
    x_train, y_train = generate_data(100, seed=42)

    # Training loop with pressure conditions (FIXED termination logic)
    best_coeffs = None
    best_mse = float("inf")

    for epoch in range(args.epoch_limit):
        elapsed = time.time() - start_time
        if elapsed >= args.time_limit:
            print(f"Time limit reached ({args.time_limit}s)")
            break

        coeffs = fit_polynomial(x_train, y_train, degree=args.degree)
        mse = compute_mse(x_train, y_train, coeffs)

        if mse < best_mse:
            best_mse = mse
            best_coeffs = coeffs

        print(f"Epoch {epoch+1}: degree={args.degree}, train_mse={mse:.6f}")

    # Save result (FIXED output format)
    result = {
        "coefficients": best_coeffs.tolist(),
        "degree": args.degree,
        "train_mse": best_mse,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Training complete. MSE={best_mse:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create evaluate.py**

```python
#!/usr/bin/env python3
"""Evaluator for autoresearch integration test.

Loads result.json, computes MSE on held-out test set.
Fixed Condition — do NOT modify this file.
"""

import json
import numpy as np


def generate_test_data(n_points=50, seed=99):
    """Generate held-out test data. FIXED."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-np.pi, np.pi, n_points)
    y = np.sin(x) + rng.normal(0, 0.05, n_points)
    return x, y


def main():
    with open("result.json") as f:
        result = json.load(f)

    coeffs = np.array(result["coefficients"])
    x_test, y_test = generate_test_data()

    y_pred = np.polyval(coeffs, x_test)
    mse = float(np.mean((y_test - y_pred) ** 2))

    print(f"mse={mse:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create autoresearch-protocol.md**

```markdown
# Autoresearch Protocol: Polynomial Sin(x) Fitting

## Research Question
Find the optimal polynomial fitting strategy to approximate y = sin(x) on [-π, π].

## Environment
- Base code: . (current directory)
- Dataset: sin(x) on [-π, π], 100 train / 50 test points, numpy generated
- Framework: numpy only

## Fixed Conditions
- Dataset generation: seed=42 for train, seed=99 for test, point counts fixed
- Termination logic: epoch_limit and time_limit in train.py (do NOT modify)
- Evaluation logic: evaluate.py (do NOT modify)
- Output format: result.json with coefficients, degree, train_mse

## Pressure Conditions
- time_limit: 10s
- epoch_limit: 1
- Whichever triggers first

## Variable Conditions
- Polynomial degree: any integer >= 1
- Feature engineering: any (e.g., Chebyshev basis, trigonometric features)
- Regularization: any (e.g., ridge via manual implementation)
- Any modification to fit_polynomial() function in train.py

## Evaluation
- metric: mse
- direction: minimize
- eval_command: "python evaluate.py"
- baseline: 0.300000

## Termination
- max_rounds: 5
- target: 0.01

## Agent Boundary
- agent_a: ["design", "code", "train"]
- agent_b: ["evaluate", "review", "record"]
```

- [ ] **Step 4: Create experiences.md**

```markdown
# Experiment Experiences

## Summary
- Best result: mse = 0.300000 (Baseline)
- Total rounds: 0 / 5
- Status: not_started
```

- [ ] **Step 5: Verify base code works**

```bash
cd tests/autoresearch/base-project
python train.py --degree 2 --epoch-limit 1 --time-limit 10
python evaluate.py
# Expected: mse≈0.3 (degree-2 polynomial, intentionally bad)
```

- [ ] **Step 6: Commit**

```bash
git add tests/autoresearch/base-project/
git commit -m "test: add autoresearch integration test base project"
```

---

### Task 6: Create Integration Test Runner and Verifier

**Files:**
- Create: `tests/autoresearch/run-test.sh`
- Create: `tests/autoresearch/verify.sh`

The test runner sets up the environment and runs Claude with the autoresearch skill. The verifier checks the results.

- [ ] **Step 1: Create run-test.sh**

```bash
#!/usr/bin/env bash
# Integration test for autoresearch skill
# Usage: ./run-test.sh [max_rounds]
#
# Sets up a temp project with polynomial fitting base code,
# runs Claude with the autoresearch skill, then verifies results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAX_ROUNDS="${1:-3}"

TIMESTAMP=$(date +%s)
TEST_PROJECT="/tmp/autoresearch-test-${TIMESTAMP}"

echo "=== Autoresearch Integration Test ==="
echo "Test project: $TEST_PROJECT"
echo "Max rounds: $MAX_ROUNDS"
echo ""

# Setup: copy base project and initialize git
mkdir -p "$TEST_PROJECT"
cp "$SCRIPT_DIR/base-project/train.py" "$TEST_PROJECT/"
cp "$SCRIPT_DIR/base-project/evaluate.py" "$TEST_PROJECT/"
cp "$SCRIPT_DIR/base-project/autoresearch-protocol.md" "$TEST_PROJECT/"
cp "$SCRIPT_DIR/base-project/experiences.md" "$TEST_PROJECT/"

# Override max_rounds in protocol if specified
if [ "$MAX_ROUNDS" != "5" ]; then
    sed -i.bak "s/max_rounds: 5/max_rounds: $MAX_ROUNDS/" "$TEST_PROJECT/autoresearch-protocol.md"
    rm -f "$TEST_PROJECT/autoresearch-protocol.md.bak"
fi

cd "$TEST_PROJECT"
git init
git add -A
git commit -m "initial: base code for autoresearch test"

# Verify base code works
echo "Verifying base code..."
python3 train.py --degree 2 --epoch-limit 1 --time-limit 10
python3 evaluate.py
echo ""

# Run Claude with autoresearch skill
PROMPT="I need you to run an automated research loop. Read autoresearch-protocol.md for the protocol, experiences.md for the current state. Use the spml:autoresearch skill."

LOG_FILE="$TEST_PROJECT/claude-output.json"

echo "Running Claude with autoresearch skill..."
echo "Plugin dir: $PLUGIN_DIR"

cd "$PLUGIN_DIR"
timeout 1800 claude -p "$PROMPT" \
    --plugin-dir "$PLUGIN_DIR" \
    --dangerously-skip-permissions \
    --max-turns 100 \
    --add-dir "$TEST_PROJECT" \
    --output-format stream-json \
    > "$LOG_FILE" 2>&1 || true

echo ""
echo "Claude finished. Running verification..."
echo ""

# Run verification
"$SCRIPT_DIR/verify.sh" "$TEST_PROJECT" "$LOG_FILE"
```

- [ ] **Step 2: Create verify.sh**

```bash
#!/usr/bin/env bash
# Verify autoresearch test results
# Usage: ./verify.sh <test-project-dir> <log-file>

set -euo pipefail

TEST_DIR="$1"
LOG_FILE="$2"

PASSED=0
FAILED=0

check() {
    local test_name="$1"
    local result="$2"
    if [ "$result" = "true" ]; then
        echo "  ✅ $test_name"
        PASSED=$((PASSED + 1))
    else
        echo "  ❌ $test_name"
        FAILED=$((FAILED + 1))
    fi
}

echo "=== Verification ==="
echo ""

# 1. Skill invocation
echo "1. Skill invocation"
SKILL_TRIGGERED=$(grep -q '"skill":"autoresearch"' "$LOG_FILE" 2>/dev/null || grep -q '"skill":"spml:autoresearch"' "$LOG_FILE" 2>/dev/null; echo $?)
check "autoresearch skill invoked" "$([ "$SKILL_TRIGGERED" = "0" ] && echo true || echo false)"

# 2. Subagents dispatched
echo "2. Subagent dispatching"
AGENT_COUNT=$(grep -c '"name":"Agent"' "$LOG_FILE" 2>/dev/null || echo "0")
check "subagents dispatched (need ≥2, got $AGENT_COUNT)" "$([ "$AGENT_COUNT" -ge 2 ] && echo true || echo false)"

# 3. Git state
echo "3. Git management"
cd "$TEST_DIR"
COMMIT_COUNT=$(git log --oneline | wc -l | tr -d ' ')
check "multiple git commits ($COMMIT_COUNT total)" "$([ "$COMMIT_COUNT" -gt 1 ] && echo true || echo false)"

# Check for autoresearch commit messages
AR_COMMITS=$(git log --oneline --grep="autoresearch" | wc -l | tr -d ' ')
check "autoresearch commits found ($AR_COMMITS)" "$([ "$AR_COMMITS" -gt 0 ] && echo true || echo false)"

# 4. experiences.md integrity
echo "4. Experience log"
if [ -f "$TEST_DIR/experiences.md" ]; then
    check "experiences.md exists" "true"

    # Check Summary section exists
    HAS_SUMMARY=$(grep -q "## Summary" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "Summary section exists" "$HAS_SUMMARY"

    # Check at least one Round entry
    ROUND_COUNT=$(grep -c "^## Round" "$TEST_DIR/experiences.md" || echo "0")
    check "round entries exist ($ROUND_COUNT)" "$([ "$ROUND_COUNT" -gt 0 ] && echo true || echo false)"

    # Check rounds have required fields
    HAS_STRATEGY=$(grep -q "Strategy" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "rounds have Strategy field" "$HAS_STRATEGY"

    HAS_RESULT=$(grep -q "Result" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "rounds have Result field" "$HAS_RESULT"

    HAS_VERDICT=$(grep -q "Verdict" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "rounds have Verdict field" "$HAS_VERDICT"

    # Check Status is not running (loop terminated)
    STATUS_DONE=$(grep -qE "Status: (completed|target_reached)" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "loop terminated (status not running)" "$STATUS_DONE"
else
    check "experiences.md exists" "false"
fi

# 5. Result quality
echo "5. Result quality"
if [ -f "$TEST_DIR/result.json" ]; then
    check "result.json exists" "true"
else
    check "result.json exists" "false"
fi

# Check final evaluation
FINAL_MSE=$(cd "$TEST_DIR" && python3 evaluate.py 2>/dev/null | sed -n 's/.*mse=\([0-9.]*\).*/\1/p' || echo "N/A")
if [ "$FINAL_MSE" != "N/A" ]; then
    check "final evaluation runs (mse=$FINAL_MSE)" "true"
    # Check if improved from baseline 0.3
    IMPROVED=$(python3 -c "print('true' if float('$FINAL_MSE') < 0.3 else 'false')" 2>/dev/null || echo "false")
    check "improved from baseline 0.3 (got $FINAL_MSE)" "$IMPROVED"
else
    check "final evaluation runs" "false"
fi

echo ""
echo "=== Summary ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "STATUS: FAILED"
    exit 1
else
    echo "STATUS: PASSED"
    exit 0
fi
```

- [ ] **Step 3: Make scripts executable and verify they parse correctly**

```bash
chmod +x tests/autoresearch/run-test.sh
chmod +x tests/autoresearch/verify.sh
bash -n tests/autoresearch/run-test.sh
bash -n tests/autoresearch/verify.sh
```

- [ ] **Step 4: Commit**

```bash
git add tests/autoresearch/run-test.sh tests/autoresearch/verify.sh
git commit -m "test: add autoresearch integration test runner and verifier"
```

---

### Task 7: Update README and Version Bump

**Files:**
- Modify: `README.md`
- Modify: `package.json`

- [ ] **Step 1: Add autoresearch to the ML Workflow section in README.md**

After the existing workflow diagram (around line 82-99), update to include the autoresearch branch:

```markdown
## The ML Workflow

```
brainstorming
    Refine hypothesis, collect context, confirm validation scope
    |
experiment-planning
    Break into atomic subtasks with validation criteria
    |
ml-subagent-dev
    Execute each subtask: unit test → implement → Validation Pyramid
    |
    ├── training-handoff (single long-running task)
    │   Generate training script + Watchdog prompt + experiment context
    │   |
    │   watchdog (independent session)
    │   Active monitoring: auto-restart, parameter fixing, anomaly diagnosis
    │
    └── autoresearch-handoff (automated iteration)
        Generate research protocol + startup prompt + experience log
        |
        autoresearch (independent session)
        Autonomous loop: Agent A designs+codes → Agent B evaluates+reviews
        Git commit on improvement, rollback on failure, experience accumulation
    |
verification
    Evidence-based conclusion: effective / ineffective / inconclusive
```
```

- [ ] **Step 2: Add autoresearch skills to the Skills table**

Add to the "ML Workflow" skills table:

```markdown
| **autoresearch-handoff** | Generate research protocol and startup prompt for autonomous iteration |
| **autoresearch** | Supervisor loop: dispatch designer/evaluator agents, manage git, accumulate experience |
```

- [ ] **Step 3: Bump version in package.json**

This is a new feature (minor bump):

```bash
# Read current version, bump minor
npm version minor --no-git-tag-version
```

- [ ] **Step 4: Commit**

```bash
git add README.md package.json
git commit -m "chore: add autoresearch to README, bump version"
```

---

### Task 8: Run Integration Test

**Files:** (no new files — execution only)

This task runs the integration test to validate the full call chain.

- [ ] **Step 1: Run the integration test with max_rounds=3**

```bash
cd tests/autoresearch
./run-test.sh 3
```

Expected: The test should complete within 30 minutes. Claude will:
1. Read the protocol
2. Run at least 2 rounds of Agent A → Agent B
3. Make git commits for improvements
4. Terminate after 3 rounds or reaching MSE ≤ 0.01

- [ ] **Step 2: Review test output**

Check the verification output for all checkpoints passing. If any fail, diagnose and fix the relevant skill file.

- [ ] **Step 3: If test passes, commit any test output fixes**

If the test revealed issues that required skill fixes, those fixes should already be committed in their respective tasks. If minor test infrastructure adjustments were needed:

```bash
git add tests/autoresearch/
git commit -m "test: fix autoresearch integration test issues"
```
