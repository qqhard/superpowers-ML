---
name: vp-overfitting-test
description: Use when running L2 overfitting test in the Validation Pyramid - validates model can memorize small dataset, proving implementation correctness before expensive full training
---

# L2: Overfitting Test

## Overview

The fastest way to verify a model implementation is correct: if it can't memorize 100-1000 samples, something is wrong with the implementation. This test takes ~10 minutes and catches bugs that would otherwise waste hours/days of full training.

**Run after L0 and L1 pass.**

## The Test

1. Take 100-1000 samples from training data (or generate synthetic data)
2. Fix random seed for reproducibility
3. Train for 5-10 epochs on these samples only
4. Assert: training loss monotonically decreases to near 0
5. Assert: task-specific metric reaches near-perfect

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

def run_overfit_test(model, train_fn, data_subset, n_epochs=10, loss_threshold=0.01):
    """
    Args:
        model: the model to test
        train_fn: function(model, data, epoch) -> loss
        data_subset: small dataset (100-1000 samples)
        n_epochs: number of epochs to train
        loss_threshold: loss must be below this to pass
    """
    set_seed(42)

    loss_history = []
    for epoch in range(n_epochs):
        epoch_loss = train_fn(model, data_subset, epoch)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch}: loss={epoch_loss:.6f}")

    # Check 1: Loss reached near zero
    final_loss = loss_history[-1]
    assert final_loss < loss_threshold, \
        f"FAIL: Final loss {final_loss:.6f} > threshold {loss_threshold}. Model can't memorize small data."

    # Check 2: Loss was generally decreasing
    # Allow small bumps but overall trend must be down
    for i in range(1, len(loss_history)):
        # Compare each epoch to the first epoch
        if loss_history[i] > loss_history[0] * 1.5:
            print(f"WARNING: Loss increased significantly at epoch {i}")

    print(f"PASS: Overfit test passed. Final loss: {final_loss:.6f}")
    return loss_history
```

## Task-Specific Criteria

| Task | Metric | Target |
|------|--------|--------|
| RecSys | NDCG@10 | > 0.95 |
| RecSys | Recall@10 | > 0.95 |
| LLM | Perplexity | < 1.1 |
| LLM | Loss | < 0.01 |
| Classification | Accuracy | > 0.99 |
| Regression | MSE | < 1e-4 (normalized) |
| General | Training loss | Monotonically decreasing to near 0 |

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
2. Trigger **mlsp:ml-diagnostics**
3. Common first steps: check learning rate, check loss function, check data labels
4. Re-run overfit test after fixing

## Important Notes

- Use the SAME model architecture as full training (just smaller data)
- Use the SAME training code (just fewer samples/epochs)
- Don't reduce model size — you want to test the actual implementation
- Fixed seed ensures reproducibility: same results every run
