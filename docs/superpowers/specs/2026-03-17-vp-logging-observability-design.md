# VP Logging & Observability Checks

**Date:** 2026-03-17
**Status:** Draft
**Scope:** Add training log output validation to L0 (static) and L1 (runtime) of the Validation Pyramid

## Problem

The Validation Pyramid currently checks engineering efficiency and training health, but does not verify that the user's training code has proper logging and observability. Without these checks, training runs may complete with no retrievable loss history, no performance metrics on file, and no way to diagnose issues post-hoc.

## Design Decisions

- **Approach:** Extend existing L0 and L1 skills (not new standalone skills)
- **Check granularity:** Semantic-level — verify that specific metrics are output, without requiring specific libraries
- **Output frequency target:** ~1 minute intervals (not every step, not long gaps)
- **Visualization tools:** User preference collected during brainstorming, not auto-detected
- **Rename:** `vp-engineering-efficiency` → `vp-static-checks` (scope now extends beyond efficiency)

## Changes

### 1. `spml:brainstorming` — New Question

Add to the ML-specific brainstorming flow:

> "Do you need visualization metrics output (e.g., WandB, TensorBoard, MLflow)? If yes, which tool do you prefer?"

Record the answer in the experiment spec. Downstream L0/L1 checks read from this spec.

### 2. `vp-engineering-efficiency` → `vp-static-checks` (L0)

Rename the skill and add section **6: Logging & Observability**.

#### New L0 Checks

| #   | Check                                        | Severity      | Static Check Method                                                                                                      |
| --- | -------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 6.1 | Loss file output                             | **Mandatory** | Code contains logic to write loss values to a file (not only stdout)                                                     |
| 6.2 | Step speed / throughput file output           | **Mandatory** | Code contains logic to write step time or throughput to a file                                                           |
| 6.3 | Data loading duration log                    | Advisory      | Code contains logic to record data loading start/end/duration                                                            |
| 6.4 | Output frequency reasonableness              | Advisory      | Log output is not triggered every step; code has interval control (estimated ~1 min)                                     |
| 6.5 | Progress bar                                 | Advisory      | Code uses a progress bar library (tqdm, rich.progress, etc.); tool not restricted                                        |
| 6.6 | Visualization tool correctness (if enabled)  | **Mandatory** | Read selected tool from experiment spec; check corresponding init and log calls exist with reasonable frequency; skip if not enabled |

**Check method:** Semantic-level. Look for file-writing patterns (`open()`/`write()`/`json.dump()`/`csv.writer`/`logging.FileHandler`/library-specific APIs) associated with the specific metric, not just generic output calls.

**Severity rules:**
- Mandatory failure → blocks progress (same as existing L0 mandatory checks)
- Advisory failure → warning, does not block
- Check 6.6 is conditional: only fires when experiment spec indicates visualization is enabled

### 3. `vp-process-metrics` (L1) — New Runtime Validation Group

Add **Logging Output Validation** to Universal Checks. Each item validates three layers: **existence → frequency → value correctness**.

| #   | Check                                          | Validation Method                                                                                                                                          |
| --- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.1 | Loss file output correctness                   | File exists, non-empty, parseable format; **values reasonable** (no all-NaN/Inf/zero, trend consistent with gradient behavior)                             |
| 3.2 | Step speed output correctness                   | File exists, non-empty; **values match wall clock** (step count × reported step time ≈ actual elapsed time)                                                |
| 3.3 | Data loading duration correctness               | Duration record exists; **values reasonable** (non-zero, non-negative, consistent with actual time window)                                                 |
| 3.4 | Output frequency reasonableness                 | Log entry interval approximately minute-level across the run                                                                                               |
| 3.5 | Progress bar correctness                        | Progress bar total matches training target — 1 epoch → total = dataset size; N steps → total = N; T minutes → time-based estimate; advance rate matches actual run speed |
| 3.6 | Visualization tool output correctness (if enabled) | Output directory has data; **frequency reasonable**; **values cross-validated against loss/speed files** (consistency check)                                |

**Relationship to L0:**
- L0 checks "code has the logic"
- L1 verifies "running code actually produces correct output"
- L1 does not repeat L0 static checks

### 4. Reference Updates (Rename Propagation)

All references to `vp-engineering-efficiency` updated to `vp-static-checks`:

| File | Type |
| --- | --- |
| `skills/vp-engineering-efficiency/` | Rename directory + SKILL.md |
| `skills/validation-pyramid/SKILL.md` | Update references |
| `skills/subagent-dev/SKILL.md` | Update references (if any) |
| `docs/superpowers/specs/2026-03-16-validation-pyramid-refactor-design.md` | Update references |
| `docs/plans/` (6 files) | Update references |

### 5. Out of Scope

- Toolkit code changes (this is about checking user code, not modifying the toolkit)
- L2 (e2e-validator) — no changes
- Existing L0/L1 checks — no modifications to current items
