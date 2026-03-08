# TDD-Validation Pyramid Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate TDD's RED-GREEN-REFACTOR rhythm into every Validation Pyramid layer, changing the relationship from "replace" to "extend".

**Architecture:** Modify existing skill files and design doc to embed TDD principles throughout the Pyramid. No new skills or files needed.

**Design doc:** `docs/plans/2026-03-08-tdd-validation-pyramid-integration-design.md`

---

## Task 1: Update design doc principles

**Files:**
- Modify: `docs/plans/2026-03-06-superpowers-ml-design.md`

**Step 1: Update Principle 3**

Change line 27 from:
```
3. Replace traditional TDD with a **Validation Pyramid** — multi-layer process metrics to separate "implementation error" from "strategy ineffective."
```
To:
```
3. Extend TDD with a **Validation Pyramid** — TDD's RED-GREEN-REFACTOR rhythm applies to every Pyramid layer. Each layer's validation is minute-level, satisfying fast feedback loops. Multi-layer process metrics separate "implementation error" from "strategy ineffective."
```

**Step 2: Update Principle 4**

Change line 28 from:
```
4. Retain function/operator-level traditional testing for deterministic code; the Validation Pyramid manages the training process.
```
To:
```
4. TDD applies at all levels: traditional unit tests for deterministic code (functions, operators), Validation Pyramid layers for non-deterministic process validation (training efficiency, convergence). The Pyramid is TDD extended to ML's non-deterministic domain.
```

**Step 3: Update Section 3.3 heading**

Change line 222 from:
```
**Replaces:** test-driven-development
```
To:
```
**Extends:** test-driven-development
```

**Step 4: Update L2 Overfitting Test criteria**

Change lines 288-289 from:
```
- RecSys: NDCG@10 / Recall@10 -> near 1.0
- LLM: perplexity < 1.1 or loss < 0.01
- Assertion: training loss must monotonically decrease to near 0
```
To:
```
- Assertion: training loss decreases steadily and quickly (monotonic decrease or consistent downward trend)
- Task-specific metrics show clear improvement trend
- No absolute threshold required — the test validates the trend, not the final value
```

**Step 5: Commit**

```bash
git add docs/plans/2026-03-06-superpowers-ml-design.md
git commit -m "docs: change TDD-Pyramid relationship from replace to extend"
```

---

## Task 2: Add TDD Rhythm to validation-pyramid SKILL.md

**Files:**
- Modify: `skills/validation-pyramid/SKILL.md`

**Step 1: Update Overview**

Change line 10 from:
```
The Validation Pyramid replaces traditional TDD for ML workflows. Instead of "write test, watch fail, write code, watch pass," it runs layered checks from cheap/fast (L0) to expensive/slow (L3), catching implementation errors before they waste GPU hours.
```
To:
```
The Validation Pyramid extends TDD to ML workflows. Each layer follows the same RED-GREEN-REFACTOR rhythm: write the validation script first, watch it fail, implement until it passes. Since every layer runs in minutes on small data, TDD's fast feedback loop applies naturally. The Pyramid runs layered checks from cheap/fast (L0) to expensive/slow (L3), catching implementation errors before they waste GPU hours.
```

**Step 2: Add TDD Rhythm section after the Orchestration Logic section (after line 59)**

Insert:
```markdown
## TDD Rhythm: RED → GREEN → REFACTOR

Every Pyramid layer follows TDD's core cycle:

### RED — Write validation script, watch it fail

Write the validation assertion BEFORE writing or optimizing implementation code. Run it. It must fail. Failure proves the validation script has discriminating power.

```python
# Example: L0 MFU check
def test_mfu_meets_target():
    result = calculate_mfu(model, input_shape, step_time)
    assert result['mfu'] >= 0.4, f"MFU {result['mfu']} below target 0.4"
# Run -> FAIL (MFU is 0.15, code not optimized yet)
```

### GREEN — Implement/optimize until validation passes

Write or optimize implementation code. Re-run validation each iteration. Multiple iterations are expected.

### REFACTOR — Clean up, keep validation passing

Clean up implementation code. Validation must stay green.

### Per-layer RED examples

| Layer | RED (write first) | GREEN (make it pass) |
|---|---|---|
| L0 Engineering Efficiency | `assert mfu >= target`, `assert fa_backend_enabled` | Optimize kernel selection, enable FA, adjust batch size |
| L1 Process Metrics | `assert no_gradient_nan()`, `assert attention_entropy > threshold` | Fix initialization, adjust lr, fix attention mask |
| L2 Overfitting Test | `assert loss_monotonically_decreasing(losses)` | Fix model/loss implementation bugs |
| L3 E2E Pipeline | `assert pipeline_completes_without_error()` | Fix data flow, shape mismatches |

### Validation passes immediately?

If the validation script passes on the first run, investigate:
- Is the threshold too lenient?
- Is the implementation already correct?
- Is the validation script actually testing what you intend?

Just like in TDD: a test passing immediately means you may not be testing the right thing. Verify the validation has discriminating power before proceeding.
```

**Step 3: Commit**

```bash
git add skills/validation-pyramid/SKILL.md
git commit -m "feat: add TDD RED-GREEN-REFACTOR rhythm to validation-pyramid"
```

---

## Task 3: Update vp-engineering-efficiency to use TDD flow

**Files:**
- Modify: `skills/vp-engineering-efficiency/SKILL.md`

**Step 1: Add TDD flow guidance after the Overview section (after line 12)**

Insert after line 12:
```markdown
## TDD Flow

Follow RED → GREEN → REFACTOR for each check:

1. **RED:** Write the assertion (e.g., `assert mfu >= target`). Run it. Confirm it fails.
2. **GREEN:** Optimize configuration/code until the assertion passes.
3. **REFACTOR:** Clean up.

If an assertion passes immediately, verify the threshold is meaningful — not too lenient.
```

**Step 2: Commit**

```bash
git add skills/vp-engineering-efficiency/SKILL.md
git commit -m "feat: add TDD flow to vp-engineering-efficiency"
```

---

## Task 4: Update vp-process-metrics to use TDD flow

**Files:**
- Modify: `skills/vp-process-metrics/SKILL.md`

**Step 1: Add TDD flow guidance after the Overview section (after line 10)**

Insert after line 10:
```markdown
## TDD Flow

Follow RED → GREEN → REFACTOR for each check:

1. **RED:** Write gradient/activation/parameter assertions and monitoring hooks. Run a few training steps. Confirm the checks catch problems (or confirm a known-broken state fails).
2. **GREEN:** Fix implementation issues (initialization, learning rate, architecture bugs) until all checks pass.
3. **REFACTOR:** Clean up monitoring code.

If all checks pass immediately on first run, verify: is the model actually training correctly, or are thresholds too lenient?
```

**Step 2: Commit**

```bash
git add skills/vp-process-metrics/SKILL.md
git commit -m "feat: add TDD flow to vp-process-metrics"
```

---

## Task 5: Update vp-overfitting-test to use TDD flow and fix criteria

**Files:**
- Modify: `skills/vp-overfitting-test/SKILL.md`

**Step 1: Add TDD flow guidance after the Overview section (after line 12)**

Insert after line 12:
```markdown
## TDD Flow

Follow RED → GREEN → REFACTOR:

1. **RED:** Write the overfit test assertion (`assert loss_monotonically_decreasing(losses)`). Run it on the current (possibly broken) implementation. Confirm it fails.
2. **GREEN:** Fix model/loss/optimizer implementation until loss decreases steadily and quickly on small data.
3. **REFACTOR:** Clean up training code.
```

**Step 2: Update The Test section**

Change line 19-20 from:
```
4. Assert: training loss monotonically decreases to near 0
5. Assert: task-specific metric reaches near-perfect
```
To:
```
4. Assert: training loss decreases steadily and quickly (consistent downward trend)
5. Assert: task-specific metric shows clear improvement trend
```

No absolute threshold required. The test validates the trend, not the final value.

**Step 3: Update implementation code**

Replace the `run_overfit_test` function (lines 36-65) with:
```python
def run_overfit_test(model, train_fn, data_subset, n_epochs=10):
    """
    Args:
        model: the model to test
        train_fn: function(model, data, epoch) -> loss
        data_subset: small dataset (100-1000 samples)
        n_epochs: number of epochs to train
    """
    set_seed(42)

    loss_history = []
    for epoch in range(n_epochs):
        epoch_loss = train_fn(model, data_subset, epoch)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch}: loss={epoch_loss:.6f}")

    # Check: Loss is decreasing steadily and quickly
    # Compare first half average to second half average
    mid = len(loss_history) // 2
    first_half_avg = sum(loss_history[:mid]) / mid
    second_half_avg = sum(loss_history[mid:]) / (len(loss_history) - mid)
    assert second_half_avg < first_half_avg, \
        f"FAIL: Loss not decreasing. First half avg: {first_half_avg:.6f}, Second half avg: {second_half_avg:.6f}"

    # Check: Loss trend is consistently downward (allow small bumps)
    decreasing_pairs = sum(1 for i in range(1, len(loss_history)) if loss_history[i] < loss_history[i-1])
    decrease_ratio = decreasing_pairs / (len(loss_history) - 1)
    assert decrease_ratio >= 0.6, \
        f"FAIL: Loss not decreasing consistently. Only {decrease_ratio:.0%} of epochs showed decrease."

    print(f"PASS: Overfit test passed. Loss decreased from {loss_history[0]:.6f} to {loss_history[-1]:.6f}")
    return loss_history
```

**Step 4: Update Task-Specific Criteria table**

Replace lines 70-78 with:
```markdown
## Task-Specific Guidance

The core criterion is always: **loss decreases steadily and quickly.** Task-specific checks are optional additional signals:

| Task | Additional Signal |
|------|------------------|
| RecSys | NDCG@10 / Recall@10 trending upward |
| LLM | Perplexity trending downward |
| Classification | Accuracy trending upward |
| Regression | MSE trending downward |
```

**Step 5: Commit**

```bash
git add skills/vp-overfitting-test/SKILL.md
git commit -m "feat: add TDD flow to vp-overfitting-test, use trend-based criteria"
```

---

## Task 6: Update vp-e2e-pipeline to use TDD flow

**Files:**
- Modify: `skills/vp-e2e-pipeline/SKILL.md`

**Step 1: Add TDD flow guidance after the Overview section (after line 10)**

Insert after line 10:
```markdown
## TDD Flow

Follow RED → GREEN → REFACTOR:

1. **RED:** Write the e2e pipeline test (data -> train -> checkpoint -> inference -> eval). Run it. Confirm it fails (pipeline not yet wired up, or integration bugs exist).
2. **GREEN:** Fix integration issues at each stage boundary until the full pipeline completes.
3. **REFACTOR:** Clean up pipeline code.
```

**Step 2: Commit**

```bash
git add skills/vp-e2e-pipeline/SKILL.md
git commit -m "feat: add TDD flow to vp-e2e-pipeline"
```

---

## Task 7: Final verification

**Step 1: Grep for remaining "replace" references**

```bash
grep -rn -i "replaces\? .*tdd\|replaces\? .*test.driven" skills/ docs/plans/2026-03-06-superpowers-ml-design.md
```

Expected: No matches. If any remain, fix them.

**Step 2: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore: clean up remaining replace-TDD references"
```
