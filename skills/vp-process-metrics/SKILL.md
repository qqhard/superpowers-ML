---
name: vp-process-metrics
description: Use when checking L1 process metrics in the Validation Pyramid - gradient health, activation distributions, parameter drift, loss spikes, and architecture-specific training signals
---

# L1: Process Metrics

## Overview

Check training process health after a few steps of actual training. This catches numerical issues, gradient problems, and architecture-specific anomalies that L0 can't detect.

**Run after L0 passes.** Typically check after the first 10-100 training steps.

## Universal Checks (always run)

### Gradient Health
See `gradient-checks.md` for detailed implementation.

- No NaN/Inf in any gradient
- Gradient norms in reasonable range (not vanishing < 1e-7, not exploding > 1e3)
- Gradient distribution not degenerate (not all zeros, not all same value)

### Parameter Drift
See `parameter-drift.md` for detailed implementation.

- Parameters changing from initialization (training is happening)
- Drift rate in expected range (not too fast = exploding, not too slow = vanishing)
- Loss spike detection (> 10x moving average)

## Architecture-Specific Checks (load based on context)

Load sub-files based on the decision tree in `validation-pyramid/decision-tree.md`:

| Architecture | Sub-file | What it checks |
|-------------|----------|----------------|
| Transformer | `activation-checks.md` | Attention distribution, attention entropy |
| Residual networks | `residual-stream.md` | Residual stream write ratio |
| MoE | `moe-checks.md` | Expert entropy, load balance |
| RecSys | `embedding-checks.md` | Embedding norm stability, negative sampling quality |
| LLM | `token-loss-checks.md` | Per-token loss distribution |
| LLM | `kv-cache-checks.md` | KV cache memory growth |

## How to Run

1. Train for 10-100 steps on real or representative data
2. During training, collect metrics using hooks (gradient, activation monitors)
3. After N steps, analyze collected metrics against thresholds
4. Report pass/fail per check

## Pass Criteria

All enabled universal + architecture-specific checks pass. Any failure -> trigger **spml:ml-diagnostics**.

## Writing Monitors

Agent writes monitoring code per-project. General pattern:

```python
# Register hooks before training
hooks = []
grad_stats = {}

def grad_hook(name):
    def hook(grad):
        grad_stats[name] = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'max': grad.abs().max().item(),
            'has_nan': torch.isnan(grad).any().item(),
            'has_inf': torch.isinf(grad).any().item(),
        }
    return hook

for name, param in model.named_parameters():
    if param.requires_grad:
        hooks.append(param.register_hook(grad_hook(name)))

# Train N steps...

# Remove hooks after collection
for h in hooks:
    h.remove()

# Analyze
for name, stats in grad_stats.items():
    assert not stats['has_nan'], f"NaN gradient in {name}"
    assert not stats['has_inf'], f"Inf gradient in {name}"
    assert stats['max'] < 1e3, f"Exploding gradient in {name}: max={stats['max']}"
    assert stats['max'] > 1e-7, f"Vanishing gradient in {name}: max={stats['max']}"
```
