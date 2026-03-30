# Autoresearch Design Spec

## Overview

Autoresearch is an automated experiment iteration system for SPML. It defines a research protocol document that drives an autonomous loop: Agent A designs strategies and modifies code, Agent B evaluates results and records experience, and a Supervisor manages the cycle, git state, and recovery.

The goal is to automate the "try → evaluate → learn → try again" cycle that researchers do manually, while maintaining fairness (pressure conditions), traceability (git + experiences.md), and isolation (worktree).

## Architecture

```
[Existing Workflow - Create Baseline]
  ml-brainstorming → experiment-planning → ml-subagent-dev → VP L1
  Produces: validated base code + autoresearch protocol definition in design doc

[Handoff]
  autoresearch-handoff: generate protocol.md + startup prompt → new session

[New Session - Execute Protocol]
  Supervisor (outer loop)
    ├── Read protocol, verify worktree + git state
    ├── Iteration 1..N
    │   ├── Agent A: read experiences + protocol → design strategy → modify code → train
    │   └── Agent B: evaluate → compare → return verdict
    │   Supervisor: if improved → git commit; if not → git rollback; update experiences.md check
    ├── Terminate: target reached or max_rounds hit
    └── Final report: best result + experience summary + git HEAD = best code
```

## Call Chain

```
ml-brainstorming
  │  Detects autoresearch scenario
  │  → writes "## Autoresearch Protocol" section in design doc
  ▼
experiment-planning                (unchanged, no autoresearch awareness)
  ▼
ml-subagent-dev → VP L1           (unchanged, no autoresearch awareness)
  │
  │  Post-Completion Gate
  │  Checks: does design doc contain "## Autoresearch Protocol"?
  │    ├── YES → present 3 options: Research / Train / Done
  │    └── NO  → present 2 options: Train / Done (current behavior)
  │
  │  User chooses "Research"
  ▼
autoresearch-handoff
  │  Verify VP + base code → generate protocol.md + prompt.md + experiences.md
  ▼ (new session)
autoresearch (supervisor)
  │  Read protocol → verify worktree + git
  │
  ├── Round 1
  │   ├── Dispatch Agent A (subagent) → design + code + train
  │   ├── Dispatch Agent B (subagent) → evaluate + review + record
  │   └── Supervisor: git commit or rollback based on verdict
  ├── Round 2..N
  │   └── (same pattern)
  │
  ├── Termination: target reached or max_rounds
  └── Final report
```

**Routing signals:**
- `ml-brainstorming` → `ml-subagent-dev`: the `## Autoresearch Protocol` section in the design doc
- `ml-subagent-dev` → `autoresearch-handoff`: user selects "Research" at Post-Completion Gate
- `autoresearch-handoff` → `autoresearch`: the `autoresearch-prompt.md` file, used in a new session

**Modified skill: `ml-subagent-dev`** — Post-Completion Gate adds a third option:

```
> All subtasks complete. VP passed. Next step:
>
> 1. **Research** — automated experiment iteration. I will invoke spml:autoresearch-handoff
>    to generate the research protocol and startup prompt for autonomous exploration.
> 2. **Train** — long-running training. I will invoke spml:training-handoff.
> 3. **Done** — experiment complete. I will invoke spml:verification.
>
> Which one?
```

Option 1 only appears when the design doc contains `## Autoresearch Protocol`.

## Skill Changes

### New: `spml:autoresearch-handoff`

Triggered after VP L1 passes when brainstorm design doc contains an `## Autoresearch Protocol` section. Replaces `training-handoff` for autoresearch scenarios.

**Checklist:**

1. **Verify VP L1 completion** — all enabled VP layers passed, record baseline metric
2. **Verify base code runs under pressure conditions** — VP L1 already does this (full pipeline: train under time/epoch limit → checkpoint → evaluation). The evaluation result becomes the protocol's baseline value
3. **Verify pressure condition termination code** — the base code must contain working time_limit / epoch_limit termination logic. This is part of Fixed Conditions and must not be modified by Agent A in subsequent iterations
4. **Generate `autoresearch-protocol.md`** — extract the 6 elements from the brainstorm design doc's Autoresearch Protocol section, fill into the protocol template
5. **Generate `autoresearch-prompt.md`** — startup prompt for the new session, containing file paths and launch instructions
6. **Initialize `experiences.md`** — empty file with Summary section header and initial baseline record
7. **Verify git state** — base code committed in worktree as the initial checkpoint
8. **Present launch instructions** — how to start the new session

**Output files (in experiment dir):**

```
<experiment-dir>/
├── autoresearch-protocol.md
├── autoresearch-prompt.md
├── experiences.md
├── plans/
│   └── ...-design.md
└── <base code files>
```

### New: `spml:autoresearch`

The Supervisor skill that runs in a new session. Manages the outer loop, dispatches agents, controls git, handles recovery.

#### Startup

1. Read `autoresearch-protocol.md` — understand all 6 elements
2. Verify worktree state: base code commit exists, `experiences.md` initialized
3. If resuming from interruption: read `experiences.md` Summary section to determine current round, read git log to verify state consistency
4. Enter main loop

#### Main Loop

```
for round in 1..max_rounds:

  1. Dispatch Agent A (subagent)
     Prompt: paths to protocol, experiences.md, worktree
     Hints: read protocol for constraints and freedom; read experiences for
            history; git log/diff shows successful code changes; do not modify
            termination or evaluation logic; write strategy to strategy.md
     Tools: file read/write, bash (for training execution)
     NO git write permissions

  2. Dispatch Agent B (subagent)
     Prompt: paths to protocol, experiences.md, strategy.md, checkpoint
     Hints: run eval_command from protocol; compare result against best in
            experiences.md Summary; write verdict + insight to experiences.md
     Tools: file read/write, bash (for evaluation execution)
     NO git write permissions
     Returns: { verdict: "improved" | "not_improved", metric_value: number }

  3. Supervisor acts on verdict
     If improved:
       - git add + git commit (strategy + code changes + updated experiences.md)
       - Update experiences.md Summary (best result, round count)
     If not_improved:
       - Preserve experiences.md (Agent B already appended the new round entry)
       - Rollback all code changes to previous commit state
       - Update experiences.md Summary (round count only)

  4. Check termination
     - metric_value meets target → terminate with "target_reached"
     - round == max_rounds → terminate with "completed"
     - Otherwise → continue

  5. Report progress
     - Current round / max_rounds
     - Current best metric + which round
     - This round's verdict
     - Trend (improving / plateauing / oscillating)
```

#### Agent A Prompt Structure

```
You are an ML researcher. Your task is to improve {metric} ({direction}).

## File Paths
- Protocol: {protocol_path}
- Experience log: {experiences_path}
- Codebase: {worktree_path}

## Instructions
- Read the protocol for research question, fixed conditions, your degrees of freedom, and pressure conditions
- Read experiences.md for all past strategies and results — decide yourself how much to read
- Git history contains successful strategies as commits — use git log / git diff to study what worked
- Do NOT modify termination logic or evaluation logic (Fixed Conditions)
- Write your strategy description to strategy.md before modifying code
- Train under pressure conditions — the code will auto-terminate at the limit
```

#### Agent B Prompt Structure

```
You are an ML experiment evaluator. Your task is to evaluate this round's result and record experience.

## File Paths
- Protocol: {protocol_path}
- Experience log: {experiences_path}
- Strategy: {strategy_path}
- Checkpoint: determined by protocol's eval_command

## Instructions
- Run the evaluation command from the protocol
- Compare the result against the best result in experiences.md Summary
- Append a new Round entry to experiences.md with: Strategy, Result, Verdict, Insight
- Return your verdict: "improved" or "not_improved" with the metric value
```

#### Anomaly Recovery

- **Agent timeout / crash**: Supervisor kills the subagent, records "agent_error" in experiences.md for this round, git rollback any partial changes, retry once. If retry also fails, skip this round and continue.
- **Training hangs**: Pressure condition termination code should prevent this. If Agent A's subagent itself hangs (not the training), Supervisor applies a timeout and kills.
- **Session interruption**: On restart, Supervisor reads experiences.md Summary to find current round and status. Git log confirms which commit is HEAD (= latest successful code). Resume from next round.
- **Consecutive failures**: If N consecutive rounds are "not_improved" (N configurable, default 5), Supervisor reports a plateau warning but continues unless max_rounds reached.

#### Final Report

When the loop terminates, Supervisor outputs:

```markdown
# Autoresearch Complete

## Result
- Best: {metric} = {value} (Round {N})
- Baseline: {metric} = {baseline_value}
- Improvement: {delta} ({percentage}%)
- Rounds: {completed} / {max_rounds}
- Termination reason: target_reached / completed

## Key Insights
<Top 3-5 insights distilled from experiences.md>

## Current State
- git HEAD = best performing code
- All experience recorded in experiences.md
- Worktree: {path}
```

### Modified: `spml:ml-brainstorming`

Add autoresearch scenario detection and protocol definition guidance.

#### Detection

During "Ask clarifying questions" phase, if the user's need matches these patterns, suggest autoresearch:

- Goal is to search/optimize rather than validate a single hypothesis
- Multiple iterative attempts expected
- "Find the best X" rather than "test whether X works"

#### Additional Questions (after detection)

Normal brainstorming questions continue (research question, architecture, dataset, etc.). Additionally guide the user to define:

1. **Fixed Conditions** — what cannot be changed?
2. **Pressure Conditions** — time limit / epoch limit per round?
3. **Variable Conditions** — what can the agent adjust?
4. **Evaluation** — metric, direction, eval command?
5. **Termination** — max rounds / target value?
6. **Agent Boundary** — A/B responsibility split? (has default, user can skip)

#### Design Doc Addition

The design doc gets an additional section:

```markdown
## Autoresearch Protocol

### Fixed Conditions
<from user>

### Pressure Conditions
- time_limit: <from user>
- epoch_limit: <from user>

### Variable Conditions
<from user>

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

#### Downstream Impact

- experiment-planning and ml-subagent-dev proceed normally to build and validate base code
- The presence of `## Autoresearch Protocol` in the design doc signals autoresearch-handoff (instead of training-handoff) after VP passes

## Protocol Document Format

`autoresearch-protocol.md`:

```markdown
# Autoresearch Protocol: <title>

## Research Question
<natural language description>

## Environment
- Base code: <path>
- Dataset: <path/description>
- Framework: <PyTorch/JAX/etc.>

## Fixed Conditions
- <condition 1>
- <condition 2>
- ...

## Pressure Conditions
- time_limit: <duration>
- epoch_limit: <count>
- Whichever triggers first

## Variable Conditions
- <dimension 1>: <range or "any">
- <dimension 2>: <range or "any">
- ...

## Evaluation
- metric: <name>
- direction: maximize / minimize
- eval_command: "<command with {checkpoint_path} placeholder>"
- baseline: <value from VP L1 run>

## Termination
- max_rounds: <N>
- target: <value, optional>

## Agent Boundary
- agent_a: ["design", "code", "train"]
- agent_b: ["evaluate", "review", "record"]
```

## Experience Log Format

`experiences.md`:

```markdown
# Experiment Experiences

## Summary
- Best result: {metric} = {value} (Round {N})
- Total rounds: {current} / {max_rounds}
- Status: running / completed / target_reached

---

## Round 1
- **Strategy**: <description>
- **Result**: {metric} = {value}
- **Verdict**: ✅ committed (abc1234)
- **Insight**: <what worked and why>

## Round 2
- **Strategy**: <description>
- **Result**: {metric} = {value}
- **Verdict**: ❌ rolled back
- **Insight**: <what didn't work and why>
```

- Summary section updated by Agent B each round, read by Supervisor for termination checks
- Round entries are append-only
- Successful rounds: git commit hash included, code details in git history
- Failed rounds: strategy + result + insight preserved for learning

## Key Design Decisions

1. **Protocol as document, not code** — the protocol is a markdown file that agents read, not executable configuration. This keeps it human-readable and editable.
2. **File paths, not injection** — agents receive file paths and tool access, not injected content. They decide what to read and how much, handling long experience logs gracefully.
3. **Git controlled by Supervisor only** — agents have no git write permissions. Supervisor commits on success, rollbacks on failure. Single point of control for code state.
4. **Pressure conditions as Fixed Conditions** — termination logic (time/epoch limits) is part of base code, validated by VP L1, and immutable across iterations. Ensures fairness.
5. **Worktree isolation** — each autoresearch runs in its own worktree, enabling multiple concurrent research tracks on the same repository.
6. **Resumable by design** — experiences.md + git log provide enough state to resume from any interruption point.

## Integration Test

Validate the full call chain with a minimal pseudo-data scenario. No GPU required, each round completes in seconds.

### Test Scenario: Polynomial Fitting

**Research question:** Find the best polynomial configuration to approximate y = sin(x) on [-π, π].

**Base code:**

- `train.py` — fits a polynomial to sin(x) sample points using numpy least squares. Takes `--epochs 1` and `--time-limit 10` as pressure condition args. Saves coefficients to `result.json`.
- `evaluate.py` — loads `result.json`, computes MSE on a held-out test set, prints `mse=<value>` to stdout.
- Termination logic: epoch limit (1 epoch) and time limit (10 seconds), whichever first.

**Protocol:**

```
Research Question: Find optimal polynomial fitting strategy for sin(x)
Fixed Conditions: dataset (sin(x) on [-π, π], 100 train / 50 test points), numpy only
Pressure Conditions: epoch_limit=1, time_limit=10s
Variable Conditions: polynomial degree (any), feature engineering (any), regularization (any)
Evaluation: metric=mse, direction=minimize, eval_command="python evaluate.py"
Termination: max_rounds=5, target=0.01
Agent Boundary: default (A=design+code+train, B=evaluate+review+record)
```

**Baseline:** degree-2 polynomial, MSE ≈ 0.3 (intentionally bad so agents can improve).

### Verification Checklist

```
1. Supervisor startup
   [ ] Reads autoresearch-protocol.md
   [ ] Verifies git state (base commit exists)
   [ ] Verifies experiences.md initialized

2. Per-round loop (at least 2 rounds)
   [ ] Agent A dispatched as subagent
   [ ] Agent A modifies code (not termination/eval logic)
   [ ] Agent A executes training
   [ ] Agent B dispatched as subagent
   [ ] Agent B runs eval_command
   [ ] Agent B writes round entry to experiences.md
   [ ] Agent B returns verdict

3. Git management
   [ ] Improved round: git commit created (verify with git log)
   [ ] Not-improved round: code rolled back to previous HEAD (verify files match)
   [ ] experiences.md preserved across rollbacks

4. Termination
   [ ] Loop terminates (target reached or max_rounds=5)
   [ ] Final report produced with best metric and round number
   [ ] git HEAD matches the best-performing commit

5. Experience log integrity
   [ ] experiences.md has one entry per round
   [ ] Summary section reflects correct best result and round count
   [ ] Successful rounds have commit hash
   [ ] Failed rounds have strategy + insight
```

### Test Structure

```
tests/autoresearch/
├── run-test.sh                    # Main test runner
├── base-project/                  # Template base code
│   ├── train.py
│   ├── evaluate.py
│   └── autoresearch-protocol.md   # Pre-filled protocol
└── verify.sh                      # Post-run verification checks
```

`run-test.sh` creates a temp directory, copies base-project, initializes git, and runs Claude with the `autoresearch` skill in headless mode. `verify.sh` parses the session transcript + git log + experiences.md to check all verification points.
