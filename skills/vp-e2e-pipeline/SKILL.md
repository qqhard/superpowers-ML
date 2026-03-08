---
name: vp-e2e-pipeline
description: Use when running L3 end-to-end pipeline test in the Validation Pyramid - validates full data-to-evaluation flow on tiny data before committing to expensive full training
---

# L3: End-to-End Pipeline Test

## Overview

Verify the complete flow works end-to-end on tiny data: data loading -> preprocessing -> training -> inference -> evaluation. This catches integration bugs between pipeline stages that unit tests and overfit tests miss.

**Run after L0, L1, and L2 pass.**

**TDD reminder:** Follow the RED → GREEN → REFACTOR rhythm defined in `validation-pyramid/SKILL.md`.

## The Test

1. Prepare tiny dataset (10-100 samples, or synthetic)
2. Run the full pipeline: data -> train (1-3 epochs) -> save checkpoint -> load checkpoint -> inference -> evaluate
3. Assert: pipeline completes without error
4. Assert: checkpoint save/load round-trips correctly
5. Assert: inference produces valid outputs (correct shape, no NaN)
6. Assert: evaluation metrics are computed (not necessarily good, just computed)

## Implementation

```python
import torch
import os
import tempfile

def run_e2e_pipeline_test(
    build_dataset_fn,
    build_model_fn,
    train_fn,
    inference_fn,
    evaluate_fn,
    n_samples=50,
    n_epochs=2,
    checkpoint_dir=None,
):
    """
    Args:
        build_dataset_fn() -> (train_data, eval_data)
        build_model_fn() -> model
        train_fn(model, train_data, n_epochs) -> model (trained)
        inference_fn(model, eval_data) -> predictions
        evaluate_fn(predictions, eval_data) -> metrics_dict
        n_samples: tiny dataset size
        n_epochs: quick training epochs
        checkpoint_dir: where to save checkpoint (temp dir if None)
    """
    if checkpoint_dir is None:
        checkpoint_dir = tempfile.mkdtemp()

    # Stage 1: Data
    print("=== Stage 1: Data ===")
    train_data, eval_data = build_dataset_fn()
    print(f"  Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

    # Stage 2: Model
    print("=== Stage 2: Model ===")
    model = build_model_fn()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Stage 3: Training
    print("=== Stage 3: Training ===")
    model = train_fn(model, train_data, n_epochs)
    print(f"  Training complete ({n_epochs} epochs)")

    # Stage 4: Checkpoint round-trip
    print("=== Stage 4: Checkpoint ===")
    ckpt_path = os.path.join(checkpoint_dir, "test_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file not created"

    model_reloaded = build_model_fn()
    model_reloaded.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # Verify weights match after round-trip
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model_reloaded.named_parameters()
    ):
        assert torch.equal(p1.data, p2.data), f"Checkpoint mismatch in {n1}"
    print("  Checkpoint save/load verified")

    # Stage 5: Inference
    print("=== Stage 5: Inference ===")
    predictions = inference_fn(model_reloaded, eval_data)
    # Check predictions are valid
    if isinstance(predictions, torch.Tensor):
        assert not torch.isnan(predictions).any(), "NaN in predictions"
        assert not torch.isinf(predictions).any(), "Inf in predictions"
        print(f"  Predictions shape: {predictions.shape}")
    else:
        print(f"  Predictions type: {type(predictions)}")

    # Stage 6: Evaluation
    print("=== Stage 6: Evaluation ===")
    metrics = evaluate_fn(predictions, eval_data)
    assert isinstance(metrics, dict), f"Expected dict metrics, got {type(metrics)}"
    assert len(metrics) > 0, "No metrics returned"
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== E2E Pipeline Test PASSED ===")
    return metrics
```

## What This Catches

| Bug | How it manifests |
|-----|-----------------|
| Data/model shape mismatch | Crash at training start |
| Checkpoint save missing keys | Crash at load |
| Eval metric wrong inputs | Crash or NaN at evaluation |
| Inference mode issues | `requires_grad` errors |
| Device mismatches | `expected cuda but got cpu` |
| Preprocessing not applied at eval | Metrics much worse than training |

## When This Test Fails

1. Identify which stage failed (data / model / train / checkpoint / inference / eval)
2. Fix the integration issue at that stage boundary
3. Re-run the full pipeline test
4. Do NOT trigger ml-diagnostics for integration bugs — this is traditional debugging

## Important Notes

- Use the SAME code paths as production (just smaller data)
- Don't mock anything — this is an integration test
- Checkpoint round-trip is critical (catches serialization bugs)
- Evaluation doesn't need to be good — just needs to complete
- Keep tiny data representative (same format, same features, same label distribution)
