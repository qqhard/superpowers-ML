# TDD-First Validation Pyramid Design

**Date:** 2026-03-08

**Summary:** Redefine the relationship between TDD and Validation Pyramid from "replace" to "extend". TDD's RED-GREEN-REFACTOR rhythm applies to every layer of the Validation Pyramid.

---

## 1. Core Idea

TDD and the Validation Pyramid are not in conflict. The Validation Pyramid is TDD extended to ML's non-deterministic domain.

- TDD's core: define "what is correct" first, watch it fail, then implement until it passes
- Pyramid's every layer is minute-level validation, satisfying TDD's fast feedback loop
- Traditional TDD covers deterministic code (functions, operators)
- Pyramid uses TDD rhythm to cover non-deterministic process (training efficiency, convergence)

**Old framing:** "Replace traditional TDD with Validation Pyramid"
**New framing:** "Extend TDD with a Validation Pyramid for ML process validation"

---

## 2. TDD Rhythm in Each Pyramid Layer

Every layer follows RED → GREEN → REFACTOR:

### RED — Write validation script first, confirm it fails

Write the validation assertion before writing/optimizing implementation code. Run it. It must fail. Failure proves the validation script has discriminating power.

### GREEN — Implement/optimize until validation passes

Iterative — multiple attempts allowed. Re-run validation script each time.

### REFACTOR — Clean up code, keep validation passing

### Per-layer examples

| Layer | RED (write first) | GREEN (make it pass) |
|---|---|---|
| L0 Engineering Efficiency | `assert mfu >= target`, `assert fa_backend_enabled` | Optimize kernel selection, enable FA, adjust batch size |
| L1 Process Metrics | `assert no_gradient_nan()`, `assert attention_entropy > threshold` | Fix initialization, adjust lr, fix attention mask |
| L2 Overfitting Test | `assert loss_monotonically_decreasing(losses)` — loss decreases steadily and quickly on small data | Fix model/loss implementation bugs |
| L3 E2E Pipeline | `assert pipeline_runs_without_error on tiny data` | Fix data flow, shape mismatches |

### Key rule: validation passes immediately?

If the validation script passes on the first run, investigate — either the threshold is too lenient, or the implementation already satisfies it. Just like in TDD, a test passing immediately means you may be testing existing behavior, not the thing you intend to build.

---

## 3. L2 Overfitting Test Clarification

L2 does NOT require loss to fall below an absolute threshold. Due to the nature of overfitting tests, we only need to observe that loss decreases steadily and quickly. Specific criteria:

- Loss monotonically decreasing over N consecutive epochs, OR
- Rate of decrease exceeds a reasonable baseline

No `loss < 0.01` assertion. The test is about the trend, not the final value.

---

## 4. Files to Modify

### 4.1 `docs/plans/2026-03-06-superpowers-ml-design.md`

- Principle 3: "Replace traditional TDD" → "Extend TDD with a Validation Pyramid"
- Principle 4: Reframe from "retain function/operator-level testing" to "TDD applies at all levels — traditional unit tests for deterministic code, Pyramid layers for process validation"
- Section 3.3 heading: "Replaces: test-driven-development" → "Extends: test-driven-development"
- L2 Overfitting Test: Remove absolute thresholds (`loss < 0.01`, `perplexity < 1.1`), replace with "loss decreases steadily and quickly"

### 4.2 `skills/validation-pyramid/SKILL.md`

- Overview: "replaces traditional TDD" → "extends TDD principles to ML process validation"
- Add new section "TDD Rhythm" explaining RED → GREEN → REFACTOR per layer
- Add rule for handling "validation passes immediately"

### 4.3 Each `vp-*` sub-skill

- Execution flow becomes: write validation script → confirm failure → write implementation → confirm pass
- `vp-overfitting-test`: threshold changed to "steady, fast decrease" instead of absolute value

### 4.4 Files NOT modified

- `skills/test-driven-development/SKILL.md` — unchanged, continues to govern traditional code TDD
- `skills/ml-brainstorming/SKILL.md` — no change needed, already references Pyramid
