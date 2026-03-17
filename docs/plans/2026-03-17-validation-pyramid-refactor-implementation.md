# Validation Pyramid Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:subagent-dev to implement this plan task-by-task. (If subagent-dev is not yet available, use spml:executing-plans.)

**Goal:** Refactor the Validation Pyramid from 4 complex layers to 3 integrated levels, embedded into the Superpowers subagent-driven-development review pipeline.

**Validation scope:** This is a skill/documentation refactor — no ML model code, no toolkit changes. Validation is manual: invoke each skill, verify it loads correctly, verify the orchestration flow description is coherent.

**Architecture:** Replace the old VP (4 layers, 20+ files, post-hoc reporting) with 3 levels (L0 static review, L1 runtime validation, L2 E2E pipeline) integrated after Superpowers code reviews in the subagent-dev workflow. Fork the Superpowers code-reviewer agent for L0. Delete old vp-* skills.

**Design doc:** `docs/superpowers/specs/2026-03-16-validation-pyramid-refactor-design.md`

---

## Shared Scaffold

### Existing infra (don't touch)

- `toolkit/profiling/` — all modules unchanged (l0_runner.py, mfu_calculator.py, dcgm_profiler.py, gap_analyzer.py, memory_profiler.py, layer_profiler.py)
- `tests/toolkit/profiling/` — all tests unchanged
- Superpowers code-reviewer at `/Users/bytedance/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.4/agents/code-reviewer.md` — read-only reference for forking

### Needs setup

- Directory `agents/` does not exist in SPML project root — create it in Task 1

---

## Task 1: Create ML Code Reviewer agent (L0)

**What:** Fork the Superpowers code-reviewer agent into SPML, add ML-specific static analysis checklist with mandatory/advisory tiers.

**Files to create:**
- `agents/ml-code-reviewer.md`

**Implementation:**

### Step 1: Create `agents/` directory and `agents/ml-code-reviewer.md`

Copy the Superpowers code-reviewer agent structure. Modify to:

1. Change frontmatter:
   - `name: ml-code-reviewer`
   - `description:` updated to reference ML static analysis

2. Keep the original 6 review areas (Plan Alignment, Code Quality, Architecture, Documentation, Issue Identification, Communication Protocol)

3. **Add Section 7: ML Static Analysis Checklist** with the full conditional checklist from the design doc. Structure:

```markdown
7. **ML Static Analysis Checklist**:

   Before reviewing code quality, check all applicable items from the ML checklist. Each check has an applicability condition — only check when the condition is met.

   **Mandatory (Critical) — must pass before proceeding:**

   | # | Check | When applicable | What to verify |
   |---|-------|----------------|---------------|
   | 1 | Device consistency | Code uses CUDA | Model, data, loss on same device |
   | 2 | Precision config | Has mixed precision / bf16 / fp16 | param.dtype matches expectation, autocast correct |
   | 3 | FlashAttention | Has Attention layers | FA available and enabled |
   | 4 | Optimizer coverage | Always | optimizer param_groups covers all trainable params |
   | 5 | LR scheduler | Has lr_scheduler | Correctly linked to optimizer |
   | 6 | DataLoader config | Has DataLoader | num_workers, pin_memory reasonable |

   Any mandatory check failure is a **Critical** issue — Implementer must fix before proceeding.

   **Advisory (Warning) — report but do not block:**

   | # | Check | When applicable | What to verify |
   |---|-------|----------------|---------------|
   | 7 | Data loading method | Dataset > 10GB or declared large | Uses mmap loading |
   | 8 | Padding waste | Variable-length sequences | Excessive padding or tail-wave waste |
   | 9 | Random seeds | Always | torch/np/random seeds set |
   | 10 | Gradient accumulation | Has gradient accumulation | accumulation_steps × micro_batch = global_batch |
   | 11 | Loss reduction | Has gradient accumulation | mean vs sum matches accumulation strategy |
   | 12 | Vocab/Embedding match | Has Embedding layer | tokenizer vocab_size == embedding dim |
   | 13 | Frozen layers | Fine-tuning scenario | Frozen layers match expectations |
   | 14 | FLOPs estimate | Always | Order-of-magnitude estimate from architecture (NOT runtime) |
   | 15 | GPU hardware info | Uses CUDA | Code's target GPU assumptions reviewed |
   | 16 | Memory estimate | Always | Param count + dtype + activations fit target GPU |
   | 17 | MoE backend | MoE architecture | Expert parallel, routing optimization, aux loss |
   | 18 | CUDA kernel selection | Uses CUDA | Optimized kernels, not fallback |

   Advisory failures are reported as **Important** or **Suggestions** — Implementer may fix or acknowledge.
```

### Step 2: Verify agent file is valid

Check that frontmatter is well-formed YAML with `name`, `description`, `model` fields.

### Step 3: Commit

```bash
git add agents/ml-code-reviewer.md
git commit -m "feat: add ml-code-reviewer agent (L0 static analysis)"
```

---

## Task 2: Create ML Code Reviewer skill (L0 skill)

**What:** Create the skill that describes when and how to invoke the ml-code-reviewer agent.

**Files to create:**
- `skills/ml-code-reviewer/SKILL.md`
- `skills/ml-code-reviewer/checklist.md`

**Implementation:**

### Step 1: Create `skills/ml-code-reviewer/SKILL.md`

```markdown
---
name: ml-code-reviewer
description: Use when reviewing ML code for static correctness — dispatched after Spec Review and Code Quality Review in the subagent-dev workflow
---

# L0: ML Static Analysis (ML Code Reviewer)

## Overview

A specialized code-reviewer subagent that checks ML-specific static correctness. Runs after standard Spec Review and Code Quality Review pass. Catches configuration errors, device mismatches, precision problems, and optimization oversights — issues that ML agents commonly introduce and that don't require runtime to detect.

**This is a RIGID skill.** Follow the checklist exactly. Don't skip applicable checks.

## When to Use

- Automatically dispatched by the orchestrator in `spml:subagent-dev` after code quality review passes
- Only for tasks that involve ML code (model, training loop, data pipeline, optimizer config)
- Skip for pure infrastructure tasks (CI, docs, config files)

## How It Works

1. Orchestrator dispatches `ml-code-reviewer` agent (defined in `agents/ml-code-reviewer.md`)
2. Agent reads all changed files
3. Agent evaluates each checklist item's applicability condition
4. For applicable items: verify the code meets the requirement
5. Report findings using Critical/Important/Suggestion severity levels

## Severity Tiers

- **Mandatory checks (1-6):** Failure is Critical — blocks progress, Implementer must fix
- **Advisory checks (7-18):** Failure is Warning — reported but does not block progress

## Fix Loop

Uses the shared fix loop from `spml:validation-pyramid`:
- Mandatory check fails → send to Implementer with specific file:line fix instructions
- Implementer fixes → re-run L0 review
- 5 consecutive failures → pause, notify user
- If fix modifies > 50 lines → rollback: re-run Spec Review + Code Quality Review + L0

## Checklist Reference

See `checklist.md` for the full conditional checklist.

## Integration

- **spml:subagent-dev** — dispatches this as a review stage
- **spml:validation-pyramid** — L0 in the 3-level pyramid
- **spml:ml-runtime-validator** — next level after L0 passes
```

### Step 2: Create `skills/ml-code-reviewer/checklist.md`

Extract the full checklist tables from the design doc (both mandatory and advisory tiers). This is the maintainable reference — add new checks here as patterns emerge.

Content: the two tables from design doc lines 71-95, with the added guidance:

```markdown
# ML Static Analysis Checklist

## How to Use

For each check: evaluate the **Condition** column against the code being reviewed. If the condition is met, verify the code satisfies **What to verify**. If not met, skip the check.

## Mandatory (Critical) — checks 1-6

Failure blocks progress. Implementer must fix before proceeding.

[table from design doc]

## Advisory (Warning) — checks 7-18

Do not block progress. Report as warnings. Implementer may fix or acknowledge and proceed.

[table from design doc]

## Adding New Checks

When a new common ML agent mistake is identified:
1. Add it to the appropriate tier (Mandatory or Advisory)
2. Define a clear applicability condition
3. Describe what to verify in specific, actionable terms
```

### Step 3: Commit

```bash
git add skills/ml-code-reviewer/
git commit -m "feat: add ml-code-reviewer skill with conditional checklist (L0)"
```

---

## Task 3: Create ML Runtime Validator skill (L1)

**What:** Create the skill that describes the minutes-level runtime validation — performance metrics, training health, timeout protection.

**Files to create:**
- `skills/ml-runtime-validator/SKILL.md`

**Implementation:**

### Step 1: Create `skills/ml-runtime-validator/SKILL.md`

Content must cover all sections from the design doc L1 section:

1. **Overview** — what this level does, when it runs (after L0 passes)
2. **Data Flow Selection** — real vs mock overfit, user declares during brainstorming
3. **Metrics Collected** — full table (Performance, Training health, Arch-specific)
4. **Failure Detection** — two tiers:
   - Project-specific baselines (from brainstorming: min MFU, max step time, min throughput)
   - Anomaly detection (always active: loss not decreasing, gradient NaN/Inf, MFU < 1%)
5. **Timeout Protection** — default 5 min runtime, timeout = runtime × 1.5, kill on timeout
6. **Toolkit Usage** — reuse `toolkit/profiling/` for performance (l0_runner.py etc.), training health written per-project
7. **Hierarchical Decomposition** — when metrics fail, decompose to find root cause
8. **Fix Loop** — shared mechanism: fail → Implementer fix → re-run → 5 failures → user. Fix > 50 lines → rollback
9. **Architecture-Specific Checks** — table mapping architecture to checks (Transformer → attention entropy, MoE → expert balance, etc.)

Frontmatter:
```yaml
---
name: ml-runtime-validator
description: Use when running L1 runtime validation — performance metrics, training health, and timeout protection during minutes-level training run
---
```

### Step 2: Commit

```bash
git add skills/ml-runtime-validator/
git commit -m "feat: add ml-runtime-validator skill (L1 runtime validation)"
```

---

## Task 4: Create ML E2E Validator skill (L2)

**What:** Create the skill for end-to-end pipeline validation — 6 stages, configurable step count, timeout protection.

**Files to create:**
- `skills/ml-e2e-validator/SKILL.md`

**Implementation:**

### Step 1: Create `skills/ml-e2e-validator/SKILL.md`

Content must cover:

1. **Overview** — verify full pipeline flow correctness, not performance
2. **6 Stages** — data loading, model instantiation, training N steps, checkpoint save/load, inference, evaluation. Default 1 step, configurable 3-5
3. **Timeout Protection** — 2 min per stage default, 10 min overall, kill on timeout
4. **Fix Loop** — shared mechanism, same as L0/L1. Fix > 50 lines → rollback
5. **Difference from L1** — comparison table (purpose, duration, what it cares about)
6. **What This Catches** — shape mismatches, device errors, serialization bugs, eval bugs, iterator exhaustion

Frontmatter:
```yaml
---
name: ml-e2e-validator
description: Use when running L2 end-to-end pipeline validation — verifies each pipeline stage runs through with 1-5 steps per stage
---
```

### Step 2: Commit

```bash
git add skills/ml-e2e-validator/
git commit -m "feat: add ml-e2e-validator skill (L2 end-to-end validation)"
```

---

## Task 5: Rewrite `validation-pyramid/SKILL.md`

**What:** Replace the old 4-layer VP orchestration with the new 3-level design. Delete `layer-overview.md` and `decision-tree.md` (content absorbed into new skills).

**Files to modify:**
- `skills/validation-pyramid/SKILL.md` — complete rewrite

**Files to delete:**
- `skills/validation-pyramid/layer-overview.md`
- `skills/validation-pyramid/decision-tree.md`

**Implementation:**

### Step 1: Rewrite `skills/validation-pyramid/SKILL.md`

New content must cover:

1. **Frontmatter** — keep name `validation-pyramid`, update description
2. **Overview** — 3-level design, integrated into subagent-dev after code reviews
3. **Architecture diagram** — the pipeline from design doc (Implementer → Spec Review → Code Quality → L0 → L1 → L2)
4. **Level summary table:**
   | Level | Skill | What it catches | Duration |
   |-------|-------|----------------|----------|
   | L0 | spml:ml-code-reviewer | Static config errors | Seconds (code review) |
   | L1 | spml:ml-runtime-validator | Performance + health anomalies | ~5 minutes |
   | L2 | spml:ml-e2e-validator | Pipeline flow errors | ~2 minutes |
5. **Shared Fix Loop** — full description with 5-retry cap, per-level counter
6. **Large Fix Rollback Rule** — > 50 lines triggers re-run of all prior stages
7. **Red Flags** — updated for new design:
   - Skipping a level because "it's probably fine"
   - Running L1 before L0 passes
   - Ignoring a failed level and proceeding
   - Not re-running after a fix
   - "I'll validate later" — validate NOW
   - Letting a timeout run instead of killing
8. **Integration** — references to brainstorming (validation scope), diagnostics (failure trigger), subagent-dev (execution)

### Step 2: Delete old helper files

```bash
rm skills/validation-pyramid/layer-overview.md
rm skills/validation-pyramid/decision-tree.md
```

### Step 3: Commit

```bash
git add skills/validation-pyramid/
git commit -m "refactor: rewrite validation-pyramid for 3-level design, remove old helper files"
```

---

## Task 6: Update `subagent-dev/SKILL.md`

**What:** Modify the subagent-dev workflow to chain L0→L1→L2 after code reviews instead of running VP during implementation.

**File to modify:**
- `skills/subagent-dev/SKILL.md`

**Implementation:**

### Step 1: Read current file

Read `skills/subagent-dev/SKILL.md` (236 lines) to understand full structure.

### Step 2: Make these specific changes

1. **Process diagram (lines 26-67):** Replace the "Validation Pyramid passed?" diamond with three sequential stages:
   ```
   Code Quality Review passed? → L0: ML Code Reviewer → L0 passed? → L1: ML Runtime Validator → L1 passed? → L2: ML E2E Validator → L2 passed? → Record conclusion
   ```
   Each "passed?" diamond has a failure path that goes to "Implementer fixes → re-run level"

2. **ML Implementer Subagent Prompt (lines 69-119):** Remove "Run Validation Pyramid" from the implementer's responsibilities (step 6, line 96). The implementer no longer runs VP — the orchestrator does after reviews.

3. **Validation Pyramid Execution section (lines 101-110):** Replace the old L0-L3 execution with new L0-L2 description:
   - L0: Dispatch `spml:ml-code-reviewer` agent
   - L1: Run `spml:ml-runtime-validator` skill (orchestrator executes)
   - L2: Run `spml:ml-e2e-validator` skill (orchestrator executes)

4. **Add fix loop description:** After each level, describe the shared fix loop:
   - Fail → resume Implementer subagent with feedback → re-run level
   - 5 failures → pause, notify user
   - Fix > 50 lines → rollback to Spec Review

5. **ML Quality Reviewer Prompt (lines 163-194):** Remove "Validation Pyramid review" section (lines 174-179) — VP is no longer the quality reviewer's concern, it's a separate stage.

6. **Integration section (lines 230-236):** Update skill references:
   - `spml:validation-pyramid` → keep (still the orchestration overview)
   - Add: `spml:ml-code-reviewer`, `spml:ml-runtime-validator`, `spml:ml-e2e-validator`
   - `spml:diagnostics` → keep

### Step 3: Commit

```bash
git add skills/subagent-dev/SKILL.md
git commit -m "refactor: integrate L0→L1→L2 validation into subagent-dev workflow"
```

---

## Task 7: Update `brainstorming/SKILL.md`

**What:** Update the validation scope section from 4 layers (L0-L3) to 3 levels (L0-L2). Add project-specific baseline prompts and L2 step count configuration.

**File to modify:**
- `skills/brainstorming/SKILL.md`

**Implementation:**

### Step 1: Read current file

Read `skills/brainstorming/SKILL.md` (157 lines).

### Step 2: Rewrite the "Confirming validation scope" section (lines 91-118)

Replace the 4-layer descriptions with:

```markdown
### Confirming validation scope

After understanding the experiment, confirm which validation levels apply:

**L0: ML Static Analysis (spml:ml-code-reviewer)**
- Always enabled for ML code tasks
- Checks: device consistency, precision, FA, optimizer, scheduler, DataLoader (mandatory); plus 12 advisory checks
- Ask: "Any project-specific checks to add?"

**L1: ML Runtime Validation (spml:ml-runtime-validator)**
- Default: enabled
- Ask: "Real data flow or mock overfit data flow?"
- Ask: "Runtime duration? (default 5 minutes)"
- Ask: "Project-specific baselines? (e.g., minimum MFU, max step time, min throughput)"
  - If user provides baselines, record them in the design doc
  - If not, L1 uses anomaly detection only

**L2: ML E2E Pipeline (spml:ml-e2e-validator)**
- Default: enabled
- Ask: "Steps per stage? (default 1, recommend 3-5 for better coverage)"

User can skip any level. Record decisions in the design doc under "Validation scope."
```

### Step 3: Commit

```bash
git add skills/brainstorming/SKILL.md
git commit -m "refactor: update brainstorming validation scope for 3-level VP"
```

---

## Task 8: Update `diagnostics/SKILL.md`

**What:** Replace old vp-* skill references with new skill names.

**File to modify:**
- `skills/diagnostics/SKILL.md`

**Implementation:**

### Step 1: Read current file

Read `skills/diagnostics/SKILL.md` (200 lines).

### Step 2: Update Related Skills section (lines 194-200)

Replace:
```
- **spml:vp-engineering-efficiency** — L0 checks that trigger Q3 diagnostics
- **spml:vp-process-metrics** — L1 checks that trigger Q1/Q2 diagnostics
- **spml:vp-overfitting-test** — L2 check that triggers Q1 diagnostics
```

With:
```
- **spml:ml-code-reviewer** — L0 static checks; failures trigger Q3 (efficiency/config) diagnostics
- **spml:ml-runtime-validator** — L1 runtime checks; failures trigger Q1 (convergence) or Q3 (efficiency) diagnostics
- **spml:ml-e2e-validator** — L2 pipeline checks; failures trigger Q2 (pipeline/data) diagnostics
```

### Step 3: Update any L0/L1/L2/L3 references in the body

Search for "L3" references and remove. Update any "L0: Engineering Efficiency" to "L0: Static Analysis", "L1: Process Metrics" to "L1: Runtime Validation", "L2: Overfitting Test" to "L2: E2E Pipeline".

### Step 4: Commit

```bash
git add skills/diagnostics/SKILL.md
git commit -m "refactor: update diagnostics skill references for new VP levels"
```

---

## Task 9: Update `experiment-planning/SKILL.md`

**What:** Update VP references in the plan template and subtask examples.

**File to modify:**
- `skills/experiment-planning/SKILL.md`

**Implementation:**

### Step 1: Read current file

Read `skills/experiment-planning/SKILL.md` (170 lines).

### Step 2: Update the subtask example "Run Validation Pyramid" section (lines 106-117)

Replace the old L0/L1/L2 example commands with new level names:

```markdown
### Step 6: Run Validation Pyramid

L0 (static) is handled by code review — no separate command.

L1 (runtime):
Run: `[project-specific training command with monitoring]`
Expected: MFU >= [baseline from brainstorm], no NaN/Inf, loss decreasing

L2 (E2E):
Run: `[project-specific E2E validation command]`
Expected: All 6 stages complete without error
```

### Step 3: Update validation scope reference in plan header (line 37)

Change "which layers enabled, which skipped, key thresholds" to "which levels enabled (L0/L1/L2), data flow choice, baselines, L2 step count".

### Step 4: Commit

```bash
git add skills/experiment-planning/SKILL.md
git commit -m "refactor: update experiment-planning VP references for new 3-level design"
```

---

## Task 10: Update `using-superpowers-ml/SKILL.md`

**What:** Update the reference to `vp-engineering-efficiency` in the skill routing guidance.

**File to modify:**
- `skills/using-superpowers-ml/SKILL.md`

**Implementation:**

### Step 1: Read current file

Read the file and find line 86: `"MFU is too low" -> diagnostics first, then vp-engineering-efficiency.`

### Step 2: Replace with new skill name

Change to: `"MFU is too low" -> diagnostics first, then ml-runtime-validator.`

### Step 3: Commit

```bash
git add skills/using-superpowers-ml/SKILL.md
git commit -m "refactor: update using-superpowers-ml VP skill reference"
```

---

## Task 11: Delete old VP skills

**What:** Remove the 4 old vp-* skill directories that have been replaced.

**Files to delete:**
- `skills/vp-engineering-efficiency/` (entire directory: SKILL.md, gpu-utilization.md, backend-checks.md, distributed-training.md)
- `skills/vp-process-metrics/` (entire directory: SKILL.md + 7 architecture-specific files)
- `skills/vp-overfitting-test/` (entire directory: SKILL.md)
- `skills/vp-e2e-pipeline/` (entire directory: SKILL.md)

**Implementation:**

### Step 1: Delete all old VP skill directories

```bash
rm -rf skills/vp-engineering-efficiency/
rm -rf skills/vp-process-metrics/
rm -rf skills/vp-overfitting-test/
rm -rf skills/vp-e2e-pipeline/
```

### Step 2: Verify no remaining references

```bash
grep -r "vp-engineering-efficiency\|vp-process-metrics\|vp-overfitting-test\|vp-e2e-pipeline" skills/
```

Expected: no matches (all references updated in Tasks 5-10).

### Step 3: Commit

```bash
git add -A skills/vp-engineering-efficiency/ skills/vp-process-metrics/ skills/vp-overfitting-test/ skills/vp-e2e-pipeline/
git commit -m "cleanup: delete old VP skills replaced by ml-code-reviewer, ml-runtime-validator, ml-e2e-validator"
```

---

## Task 12: Final cross-reference verification

**What:** Verify all skill references are consistent across the entire SPML project.

**Implementation:**

### Step 1: Grep for any remaining old VP references

```bash
grep -r "vp-engineering-efficiency\|vp-process-metrics\|vp-overfitting-test\|vp-e2e-pipeline\|L3.*pipeline\|L3.*e2e" skills/ agents/ docs/
```

Expected: no matches except in the design doc (which documents the old system for historical context).

### Step 2: Verify new skill references resolve

Check that all `spml:ml-code-reviewer`, `spml:ml-runtime-validator`, `spml:ml-e2e-validator` references point to existing files:

```bash
ls skills/ml-code-reviewer/SKILL.md
ls skills/ml-runtime-validator/SKILL.md
ls skills/ml-e2e-validator/SKILL.md
ls agents/ml-code-reviewer.md
```

### Step 3: Verify no broken cross-references in validation-pyramid

Read `skills/validation-pyramid/SKILL.md` and verify all referenced skills exist.

### Step 4: Commit (if any fixes needed)

```bash
git commit -m "fix: resolve any remaining cross-reference issues"
```
