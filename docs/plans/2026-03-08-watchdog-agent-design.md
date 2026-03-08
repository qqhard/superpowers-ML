# Watchdog Agent Design Document

> **For Claude:** This is a design document for the Watchdog Agent system. Use superpowers:writing-plans to create the implementation plan.

**Project:** mlsp — Watchdog Agent for long-running ML tasks

**Date:** 2026-03-08

---

## 1. Problem Definition

ML workflow differs fundamentally from traditional software engineering: after code passes Validation Pyramid checks (minute-level), there is still a **long-running execution phase** (training, data processing, inference) that IS the production output. The current workflow ends after VP passes, leaving this gap unaddressed.

Traditional software: code + tests = fast feedback; long-running production is separate from the agent's scope.

ML: the long-running process itself is production. Someone needs to watch it.

---

## 2. Solution: Session Chain

Connect multiple independent agent sessions through prompt + file passing. Each session has a single responsibility and clean context.

```
Session 1: Main Agent (brainstorm → plan → execute → VP)
    │
    │ Produces: training script + structured logs + experiment-context.md + Watchdog prompt
    │
    ▼
Session 2: Watchdog Agent (user pastes Watchdog prompt)
    │ Read-only monitoring, adaptive frequency, user can intervene anytime
    │
    │ Issue found → updates experiment-context.md + produces recovery-prompt.md
    │        or
    │ Normal completion → updates experiment-context.md + produces completion-prompt.md
    │
    ▼
Session 3: Recovery/Completion Agent (user pastes corresponding prompt)
    │ Reads experiment-context.md, autonomously decides which workflow stage to return to
    │
    ▼
  (May produce new training script + Watchdog prompt → cycle repeats)
```

### Design Principles

1. **Training script is independent of agent ecosystem** — deployable to production as-is, no agentic framework dependency
2. **Watchdog prompt is framework-agnostic** — works with Claude Code, Codex, Cursor, or any LLM agent
3. **Read-only monitoring** — Watchdog observes, diagnoses, packages context; never intervenes
4. **Files are context carriers** — prompt is the entry (short), `experiment-context.md` is the full context (long)
5. **User chooses execution mode** — separated (training and monitoring decoupled) or combined (Watchdog launches training)

---

## 3. Three New Skills

### 3.1 ml-training-handoff

**Trigger:** After ml-subagent-dev completes all subtasks and VP passes, when the task includes a long-running phase.

**Produces:**

| Artifact | Description |
|----------|-------------|
| Training script | `run_training.sh` / `run_training.py` — core code, zero agent dependency, production-deployable |
| Structured logging | Integrated in training code — dual output (tqdm progress bar for terminal + JSONL for Watchdog) |
| experiment-context.md | Full experiment context: design, VP baseline, training config, code state |
| watchdog-prompt.md | Short prompt for starting a Watchdog session |

**Dual output logging:**

- Terminal (for humans): tqdm progress bar, one line continuously updating
  ```
  Training: 10%|████▌                                    | 1000/10000 [32:15<4:50:15, loss=0.823, MFU=45%]
  ```
- JSONL file (for Watchdog): detailed per-step metrics including all indicators confirmed during VP L1
  ```json
  {"step":1000,"loss":0.823,"grad_norm":1.42,"lr":0.0001,"mfu":0.45,"memory_mb":24531,"timestamp":"2026-03-08T14:32:15"}
  ```

**User options at handoff:**

```
Option A: Separated execution
  1. Launch training: bash run_training.sh
  2. Open a new agent session, paste contents of watchdog-prompt.md

Option B: Combined execution
  1. Open a new agent session, paste contents of watchdog-prompt.md
  2. Let Watchdog launch training and monitor within the same session
```

### 3.2 ml-watchdog

**Role:** Watchdog Agent behavior definition. After user pastes Watchdog prompt into a new session, the agent follows this skill.

**Startup:**

1. Read `experiment-context.md` — understand experiment background, VP baseline, expected behavior
2. Read `metrics.jsonl` — confirm log exists and format is correct
3. Confirm training process is running (or ask user if they want to launch within this session)
4. Enter monitoring loop

**Monitoring loop:**

```
loop {
    1. Read latest N records from metrics.jsonl
    2. Analyze:
       - Compare against VP baseline (recorded in experiment-context.md)
       - Trend detection (loss trajectory, gradient norm stability)
       - Anomaly identification (spike, NaN, stagnation, sudden change)
    3. Decide:
       - Normal → continue (silent in separated mode; report progress in combined mode)
       - Mild anomaly → log, increase check frequency, continue observing
       - Severe anomaly → stop loop, enter diagnosis mode
       - Training complete → stop loop, enter completion mode
    4. Wait (adaptive interval)
}
```

**Adaptive frequency:**

| Phase | Frequency | Rationale |
|-------|-----------|-----------|
| First 10% of steps | High (every 1-2 min) | Catch startup issues |
| 10%-90% of steps | Normal (every 5-10 min) | Steady-state monitoring |
| Last 10% of steps | Higher | Watch final convergence |
| After mild anomaly | Temporarily high | Close observation |
| User override | Anytime | User can say "check now" or "change to every 3 min" |

**Output behavior depends on execution mode:**

- **Separated mode:** Silent when normal. Only outputs when anomaly detected.
- **Combined mode:** Periodic progress reports (since this is the user's only window into training status) + anomaly diagnostics when issues found.

**Diagnosis mode (on severe anomaly):**

1. Collect evidence from JSONL (metrics before/after anomaly, which metrics co-moved, timing)
2. Write diagnosis report into `experiment-context.md` (Watchdog Status + Diagnosis History sections)
3. Produce `recovery-prompt.md`
4. Notify user: "Anomaly detected at step N. Diagnosis written. Paste recovery-prompt.md into a new session."

**Completion mode (training finishes normally):**

1. Final metrics summary
2. Compare long-training results against VP baseline
3. Update `experiment-context.md`
4. Produce `completion-prompt.md`
5. Notify user: "Training complete. Paste completion-prompt.md into a new session for analysis."

**Watchdog boundaries:**

- **Does:** read logs, analyze trends, write diagnosis reports, produce prompts, notify user
- **Does not:** stop training, modify code, adjust hyperparameters, rollback checkpoints

### 3.3 ml-training-resume

**Role:** Recovery or Completion Agent entry point. User pastes recovery-prompt.md or completion-prompt.md.

**Startup:**

1. Read `experiment-context.md` — full context including Watchdog diagnosis
2. Read `metrics.jsonl` if deeper analysis needed
3. Autonomously decide next step

**Anomaly recovery (from recovery-prompt.md):**

Agent judges which workflow stage to return to based on evidence:

| Problem Layer | Return To |
|---------------|-----------|
| Code bug (NaN, shape error) | Fix code → re-run VP → re-handoff |
| Hyperparameters wrong (not converging, code is correct) | Adjust hyperparams → possibly re-run VP L2 → re-handoff |
| Task decomposition wrong (direction is off) | ml-experiment-planning |
| Hypothesis itself is wrong | ml-brainstorming |
| Data quality issue | ml-data-preparation |

Agent decides based on evidence in experiment-context.md. No pre-prescribed rollback suggestion from Watchdog.

**Normal completion (from completion-prompt.md):**

1. Analyze final training results vs experiment hypothesis
2. Compare VP baseline vs actual long-training performance
3. Enter ml-verification flow:
   - Is the conclusion supported by evidence?
   - Was the hypothesis verified/refuted?
   - Produce experiment summary report
4. Present decision options to user (same as existing ml-verification):
   - Finish → finishing-a-development-branch
   - Add subtasks → continue experiment
   - New brainstorm → new direction

**Key constraints:**

- Resume Agent does NOT assume where the problem is — judges from evidence
- All judgment basis comes from files (experiment-context.md + metrics.jsonl), not user narration
- If Agent cannot determine, it asks user for additional info
- If fix requires another long-running phase → goes through handoff → Watchdog cycle again

---

## 4. experiment-context.md Lifecycle

This file is the core link across the session chain.

### Created (Handoff)

Main Agent generates initial version:

```markdown
# Experiment Context: [name]

## Experiment Design
- Hypothesis: ...
- Independent/Dependent/Control variables: ...

## VP Baseline
- MFU: 0.45, Gradient norm: 0.1-8.0, Initial loss: 2.3
- Overfit test: loss → 0.001 in 8 epochs

## Training Configuration
- Script: run_training.sh
- Log: logs/metrics.jsonl
- Checkpoints: checkpoints/
- Expected: ~10000 steps, ~4 hours

## Code State
- Git commit: abc1234
- Branch: experiment/attention-scaling

## Watchdog Status
- Status: not started

## Diagnosis History
(empty)
```

### Updated (Watchdog)

Watchdog continuously updates `Watchdog Status` and `Diagnosis History`:

```markdown
## Watchdog Status
- Status: monitoring
- Last check: step 3200, 2026-03-08 14:45
- Mode: high-frequency (anomaly detected)

## Diagnosis History
### Issue #1 — step 3200
- Symptom: loss spike (0.48 → 1.85), grad_norm spike (2.1 → 45.3)
- Context: LR constant at this stage, not a schedule transition
- 300 steps without recovery
- Saved checkpoints: step_3000 (last good), step_3200 (onset)
- Recovery prompt: recovery-prompt.md
```

### Consumed (Resume)

Recovery Agent reads the entire file. Has the complete chain from experiment design to anomaly diagnosis.

### Appended (Multi-round)

If fix requires re-training, handoff **appends** a new round rather than overwriting, preserving full history:

```markdown
## Round 2
### Code Fix
- Fixed: data loader bug causing corrupted batch at epoch boundary
- Git commit: def5678

### VP Re-validation
- L0 ✅ L1 ✅ L2 ✅

### Training Configuration (Round 2)
- Resumed from: checkpoints/step_3000
- ...

### Watchdog Status (Round 2)
- Status: not started
```

---

## 5. Impact on Existing Workflow and Design

### Workflow Change

```
Before:
brainstorm → plan → execute + VP → ml-verification → done

After:
brainstorm → plan → execute + VP → handoff → [long-running] → watchdog → resume → ml-verification → done
                                                                              ↑
                                                         resume may return to brainstorm/plan/execute
```

### Existing Skills: No Modifications Needed

The three new skills (handoff, watchdog, resume) exist as independent workflow steps. Existing skills (ml-subagent-dev, ml-verification, ml-brainstorming, ml-data-preparation) do NOT need to "sense" the long-running phase — ML tasks are almost always long-running, so this is the default assumption. Routing to handoff is determined at the plan level, not embedded in each skill.

### Phase Delivery

Belongs in **Phase 3 (Complete Workflow)** — the goal is end-to-end workflow closure, and these three skills fill the missing segment.

Updated Phase 3 scope:
- vp-e2e-pipeline skill
- ml-diagnostics skill
- ml-subagent-dev skill
- ml-verification skill
- ml-training-handoff skill (new)
- ml-watchdog skill (new)
- ml-training-resume skill (new)

---

## 6. Scope: General Long-Running Task Monitor

Watchdog is NOT limited to training. It monitors any long-running ML task:

- Training (most common)
- Full-scale data processing (from ml-data-preparation)
- Large-scale inference / evaluation

The monitoring logic adapts based on the task type recorded in experiment-context.md. The core mechanism (read structured logs → analyze → diagnose → produce prompt) is the same regardless of task type.

---

## 7. Simplification Decisions

| Decision | Rationale |
|----------|-----------|
| No auto-intervention by Watchdog | "Watchdog" means watch, not act. Keeps the boundary clean and safe. |
| No formal schema for experiment-context.md | Agent generates and consumes it. Natural language sections are flexible and sufficient. |
| No daemon/script-level watchdog | Using an LLM agent for monitoring provides understanding, not just threshold detection. Framework-agnostic. |
| Watchdog doesn't suggest rollback level | Resume Agent has full context and can judge better. Avoids Watchdog overstepping its role. |
| No auto-start of sessions | User manually starts each session with a prompt. Keeps the system independent of any specific agentic framework. |
| No modifications to existing skills | Existing skills don't need to "sense" the long-running phase. ML tasks are almost always long-running — handoff is the default next step after VP, determined at the plan level. |
