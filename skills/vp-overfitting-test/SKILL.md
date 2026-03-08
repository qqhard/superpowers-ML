---
name: vp-overfitting-test
description: Use when running L2 overfitting test in the Validation Pyramid - validates model can memorize small dataset, proving implementation correctness before expensive full training
---

# L2: Overfitting Test

## Overview

The fastest way to verify a model implementation is correct: if it can't memorize 100-1000 samples, something is wrong with the implementation. This test takes ~10 minutes and catches bugs that would otherwise waste hours/days of full training.

**Run after L0 and L1 pass.**

## TDD Flow

Follow RED → GREEN → REFACTOR:

1. **RED:** Write the overfit test assertion (`assert loss_monotonically_decreasing(losses)`). Run it on the current (possibly broken) implementation. Confirm it fails.
2. **GREEN:** Fix model/loss/optimizer implementation until loss decreases steadily and quickly on small data.
3. **REFACTOR:** Clean up training code.

## The Test

1. Take 100-1000 samples from training data (or generate synthetic data)
2. Fix random seed for reproducibility
3. Train for 5-10 epochs on these samples only
4. Assert: training loss decreases steadily and quickly (consistent downward trend)
5. Assert: task-specific metric shows clear improvement trend

## Implementation

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

## Task-Specific Guidance

The core criterion is always: **loss decreases steadily and quickly.** Task-specific checks are optional additional signals:

| Task | Additional Signal |
|------|------------------|
| RecSys | NDCG@10 / Recall@10 trending upward |
| LLM | Perplexity trending downward |
| Classification | Accuracy trending upward |
| Regression | MSE trending downward |

## Common Failure Causes

| Symptom | Likely Cause |
|---------|-------------|
| Loss doesn't decrease at all | Learning rate too low, optimizer not stepping, gradients zero |
| Loss decreases then plateaus high | Model capacity too low for even small data, or data labels wrong |
| Loss oscillates wildly | Learning rate too high, numerical instability |
| Loss goes to NaN | Numerical overflow, check L1 (gradient/activation checks) |
| Loss reaches 0 but metric stays low | Loss and metric measure different things, check evaluation code |

## When This Test Fails

1. Do NOT proceed to L3 or full training
2. Trigger **spml:ml-diagnostics**
3. Common first steps: check learning rate, check loss function, check data labels
4. Re-run overfit test after fixing

## Important Notes

- Use the SAME model architecture as full training (just smaller data)
- Use the SAME training code (just fewer samples/epochs)
- Don't reduce model size — you want to test the actual implementation
- Fixed seed ensures reproducibility: same results every run
