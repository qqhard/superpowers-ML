> **Note:** L2/ml-e2e-validator references in this document are stale — L2 has been merged into L1 as of v0.9.0. See `docs/superpowers/specs/2026-03-29-vp-l1-l2-merge-design.md`.

# VP Logging & Observability Checks

**Date:** 2026-03-17
**Status:** Implemented
**Scope:** Add training and evaluation log output validation to L0 (static) and L1 (runtime) of the Validation Pyramid

## Problem

The Validation Pyramid currently checks ML code correctness and training health, but does not verify that the user's training and evaluation code has proper logging and observability. Without these checks, runs may complete with no retrievable loss history, no visible evaluation progress, no performance metrics on file, and no way to diagnose issues post-hoc.

## Design Decisions

- **Approach:** Extend existing L0 and L1 skills (not new standalone skills)
- **Check granularity:** Semantic-level — verify that specific metrics are output, without requiring specific libraries
- **Output frequency target:** ~1 minute intervals (not every step, not long gaps)
- **Visualization tools:** User preference collected during brainstorming, not auto-detected
- **Rename:** `ml-code-reviewer` → `ml-static-checks` (scope now extends beyond ML code review to include observability)

## Changes

### 1. `spml:ml-brainstorming` — New Question

Add to the ML-specific brainstorming flow (in the "Confirming validation scope" section):

> "Do you need visualization metrics output (e.g., WandB, TensorBoard, MLflow)? If yes, which tool do you prefer?"

Record the answer in the experiment design doc. Downstream L0/L1 checks read from this doc.

### 2. `ml-code-reviewer` → `ml-static-checks` (L0)

Rename the skill directory and add new checks to `checklist.md`, continuing the existing flat numbering (19-24).

#### New L0 Checks

Mandatory checks (19, 20, 24) are added to the existing **Mandatory (Critical)** table. Advisory checks (21, 22, 23) are added to the existing **Advisory (Warning)** table.

| #  | Check                                       | Severity          | Condition       | What to verify                                                                                                                     |
| -- | ------------------------------------------- | ----------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 19 | Loss file output                            | **Mandatory**     | Always          | Code contains logic to write loss values to a file (not only stdout) — look for file-writing patterns associated with loss values  |
| 20 | Step speed / throughput file output          | **Mandatory**     | Always          | Code contains logic to write step time or throughput to a file                                                                     |
| 21 | Data loading duration log                   | Advisory          | Has DataLoader  | Code records data loading start/end/duration                                                                                       |
| 22 | Output frequency control                    | Advisory          | Has file logging | Log output has interval control in code (e.g., `if step % N == 0`, time-based gating); not triggered every step                   |
| 23 | Progress bar                                | Advisory          | Always          | Code uses a progress bar library (tqdm, rich.progress, etc.); tool not restricted                                                  |
| 24 | Visualization tool correctness (if enabled) | **Mandatory**     | Enabled in experiment design doc | Read selected tool from design doc; check corresponding init, log calls, and frequency control exist; skip if not enabled |

**Check method:** Semantic-level. Look for file-writing patterns (`open()`/`write()`/`json.dump()`/`csv.writer`/`logging.FileHandler`/library-specific APIs) associated with the specific metric, not just generic output calls.

**Severity rules:**
- Mandatory failure → blocks progress (same as existing mandatory checks 1-6)
- Advisory failure → warning, does not block
- Check 24 is conditional: only fires when experiment design doc indicates visualization is enabled

### 3. `ml-runtime-validator` (L1) — New Logging Output Validation Section

Add a new **Logging Output Validation** section as a third category alongside the existing Performance and Training health sections in `ml-runtime-validator/SKILL.md`. Checks are prefixed with `L.` to distinguish from the existing metric categories. Each check validates three layers: **existence → frequency → value correctness** for both training output and evaluation-phase output where applicable.

| #   | Check                                              | Severity          | Validation Method                                                                                                                                          |
| --- | -------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L.1 | Loss file output correctness                       | **Mandatory**     | File exists and non-empty; parseable structured format (JSON, CSV, or other); **values reasonable** (no all-NaN/Inf/zero, trend consistent with gradient behavior) |
| L.2 | Step speed output correctness                      | **Mandatory**     | File exists and non-empty; **values match wall clock** (step count × reported step time ≈ actual elapsed time)                                              |
| L.3 | Data loading duration correctness                  | Advisory          | Duration record exists; **values reasonable** (non-zero, non-negative, consistent with actual time window)                                                  |
| L.4 | Output frequency reasonableness                    | Advisory          | Actual log entry timestamps across the run have intervals approximately minute-level (contrasts with L0 check 22 which checks for interval-control logic in code) |
| L.5 | Progress bar correctness                           | Advisory          | Progress bar total matches training target — 1 epoch → total = dataset size; N steps → total = N; T minutes → time-based estimate; advance rate matches actual run speed |
| L.6 | Visualization tool output correctness (if enabled) | **Mandatory**     | Output directory/API has data; **frequency reasonable**; **values cross-validated against loss/speed files** for consistency; skip if not enabled            |

**Evaluation-phase expectations:** When evaluation exists, L1 should also verify:
- explicit evaluation phase start/end markers
- dedicated evaluation progress output
- no long silent gaps during long-running evaluation
- checkpoint-based evaluation reports checkpoint load latency/behavior
- in-training evaluation reports that it is using in-memory state
- in-training evaluation fires at the planned step cadence
- evaluation emits timing/throughput summaries after completion

**Three-layer validation per check:** Each check internally verifies (1) the output **exists**, (2) the output **frequency** is reasonable, and (3) the output **values are correct**. The table describes the combined criteria. Check L.4 is the aggregate frequency check across all log outputs.

**L0 vs L1 distinction:**
- L0 checks "code has the logic" (static analysis of source files)
- L1 verifies "running code actually produces correct output" (inspect files/stdout after training and evaluation steps)
- L0 check 22 verifies interval-control logic exists in code; L1 check L.4 verifies actual timestamps are minute-level — they are complementary, not duplicative

### 4. Reference Updates (Rename Propagation)

Rename `ml-code-reviewer` → `ml-static-checks`. Files to update:

| File | Change |
| --- | --- |
| `skills/ml-code-reviewer/` | Rename directory to `skills/ml-static-checks/` |
| `skills/ml-code-reviewer/SKILL.md` | Update name, title, description |
| `skills/ml-code-reviewer/checklist.md` | Add checks 19-24 |
| `skills/ml-runtime-validator/SKILL.md` | Update reference from `ml-code-reviewer` to `ml-static-checks` |
| `skills/validation-pyramid/SKILL.md` | Update L0 reference |
| `skills/subagent-dev/SKILL.md` | Update L0 references |
| `skills/using-superpowers-ml/SKILL.md` | Update L0 reference |
| `skills/brainstorming/SKILL.md` | Update L0 reference |
| `skills/diagnostics/SKILL.md` | Update L0 reference |
| `README.md` | Update L0 reference |
| `agents/ml-code-reviewer.md` | Rename to `agents/ml-static-checks.md`, update contents |
| `.claude-plugin/plugin.json` or skill registration | Update skill name mapping |
| Plugin cache | Reinstall/resync after source changes |

Historical plan documents (`docs/plans/`) are not updated — they reflect the state at time of writing.

### 5. Out of Scope

- Toolkit code changes (this is about checking user code, not modifying the toolkit)
- L2 (ml-e2e-validator) — no changes
- Existing L0 checks 1-18 — no modifications
- Existing L1 Performance/Training health metrics — no modifications
