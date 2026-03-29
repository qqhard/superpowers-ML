# Validation Pyramid L1/L2 Merge Design

**Date:** 2026-03-29
**Status:** Draft

## Summary

Merge the Validation Pyramid from three levels (L0 Static, L1 Runtime, L2 E2E) into two levels (L0 Static, L1 Runtime). The new L1 runs a sequential pipeline: train for an estimated 5 minutes of training volume (collecting performance and health metrics), then verify the full pipeline (checkpoint, inference, evaluation). One run, one unified report.

## Motivation

L1 and L2 are both runtime validation — they both execute code and overlap in training steps, loss checks, and gradient checks. Merging simplifies the pyramid, reduces total validation time (no duplicate training), and produces a single coherent report instead of two.

## Architecture

### VP Structure (Before → After)

**Before (3 levels):**

| Level | Name | Duration |
|-------|------|----------|
| L0 | ML Static Checks | Seconds |
| L1 | ML Runtime Validator | ~5-7 min |
| L2 | ML E2E Validator | ~2-5 min |

**After (2 levels):**

| Level | Name | Duration |
|-------|------|----------|
| L0 | ML Static Checks | Seconds |
| L1 | ML Runtime Validation | ~5-15 min |

### New L1 Pipeline (Sequential)

```
Stage 1: Data Loading
  - Load N batches, verify shapes, no NaN/Inf in inputs

Stage 2: Model Instantiation
  - Create model, accept input, verify output shape

Stage 3: Training (~5 min estimated training volume)
  - forward + backward + optimizer.step
  - Collect simultaneously:
    · Performance: MFU, TCA, throughput, memory
    · Health: loss trend, gradient health, parameter drift
    · Architecture-specific: attention entropy / expert balance / embedding stability
    · Logging: loss file, step speed file, visualization tool output

Stage 4: Checkpoint
  - save → load into fresh instance → params match

Stage 5: Inference
  - eval mode, N forward passes, no NaN/Inf, deterministic

Stage 6: Evaluation
  - Compute metrics on N batches, results finite, function doesn't crash
```

### Training Volume Control

Priority order (unchanged from original L1):

1. Config/CLI override (`--max_steps=N`)
2. L1-specific config file
3. Wrapper script

**Forbidden:** Monkey-patching runtime objects.

The target is an estimated 5 minutes of training volume. This is the default; users can customize during brainstorming.

### Timeout Protection

- **Total timeout:** 15 minutes
- **Background execution** with 30-second liveness checks
- Timeout = bug to investigate (training volume estimation was wrong, or a stage hung)

### Failure Detection (Two Tiers)

1. **Project baselines** (defined in brainstorm): minimum MFU, maximum step time, minimum throughput
2. **Anomaly detection** (always active):
   - Loss not decreasing or increasing
   - Gradients all NaN/Inf
   - MFU < 1%
   - Memory fragmentation extreme
   - Logging outputs missing/empty
   - Pipeline stage crash or timeout

### Fix Loop

Identical to L0 — failure → Implementer fixes → re-run entire L1 (from stage 1) → 5 consecutive failures → pause for user intervention.

**Large fix rollback:** If fix modifies > 50 lines, roll back and re-run Spec Review + Code Quality Review + L0 + L1.

### Output Report (Unified)

```
=== L1 Runtime Validation Report ===

[Data Loading]     PASS  5 batches loaded, shapes correct
[Model Init]       PASS  output shape [B, C] matches expected
[Training]         PASS  150 steps, 4m52s
  MFU:             38.2%
  Throughput:      12.4k tokens/sec
  Peak Memory:     24.3 GB
  Loss:            2.41 -> 1.87 (down 23%)
  Gradients:       healthy (no NaN/Inf)
  Param Drift:     normal
[Checkpoint]       PASS  save/load match (max diff: 1e-7)
[Inference]        PASS  5 passes, deterministic
[Evaluation]       PASS  metrics computed (acc: 0.34)

RESULT: PASS (6/6 stages)
Total time: 5m38s
```

## Orchestration Changes

### ml-subagent-dev Flow

```
Before: Implementer → L0 → L1 → L2 → Spec Review → Quality Review → Conclusion
After:  Implementer → L0 → L1 → Spec Review → Quality Review → Conclusion
```

### Completion Gate (5 items, was 6)

- [ ] L0: Static Checks — passed
- [ ] L1: Runtime Validation — passed
- [ ] Spec Review — passed
- [ ] Quality Review — passed
- [ ] Conclusion recorded — with metric evidence

## Upstream/Downstream Changes

### ml-brainstorming

- Remove "L2 enabled/disabled" option
- VP config: L0 enabled/disabled + L1 enabled/disabled
- Add question: "How much training volume? (default ~5 minutes)"
- Project baseline config unchanged (min MFU, max step time, etc.)

### experiment-planning

- Subtask VP levels: `L0/L1` (was `L0/L1/L2`)

### verification

- Remove separate L2 check; confirm L1 report covers full pipeline

### using-spml skill routing

- Remove `spml:ml-e2e-validator` entry
- Update `spml:ml-runtime-validator` description: "Runtime validation — training + full pipeline E2E"

### diagnostics / training-handoff

- Update any L2 references to L1

## File Changes

### Delete

- `skills/ml-e2e-validator/SKILL.md`

### Rewrite

- `skills/ml-runtime-validator/SKILL.md` — merged L1 spec

### Modify

- `skills/validation-pyramid/SKILL.md` — 3-level → 2-level
- `skills/ml-subagent-dev/SKILL.md` — orchestration flow + completion gate
- `skills/ml-brainstorming/SKILL.md` — VP config section
- `skills/experiment-planning/SKILL.md` — subtask VP levels
- `skills/verification/SKILL.md` — final checklist
- `skills/training-handoff/SKILL.md` — VP references
- `skills/diagnostics/SKILL.md` — VP references (if any)
- `skills/using-superpowers-ml/SKILL.md` — skill routing table

### Unchanged

- `skills/ml-static-checks/SKILL.md` — L0 unchanged
- `agents/ml-static-checks.md` — L0 agent unchanged
- `toolkit/profiling/l0_runner.py` — toolkit unchanged
