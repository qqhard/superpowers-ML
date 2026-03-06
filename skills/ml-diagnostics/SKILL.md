---
name: ml-diagnostics
description: Use when ML training isn't converging, shows anomalous behavior, or runs too slowly - systematic diagnosis for training failures, early anomalies, and efficiency bottlenecks
---

# ML Diagnostics

## Overview

Systematic diagnosis for ML training problems. Replaces random "try this" debugging with structured evidence gathering and hypothesis testing.

**Core principle:** In ML, "not working" is normal. The question is WHY it's not working — implementation bug or strategy limitation? Misdiagnosing this wastes entire research directions.

<HARD-GATE>
Do NOT propose fixes without completing Phase 1 (evidence gathering). The temptation to "just try lowering the learning rate" is the #1 cause of wasted GPU hours.
</HARD-GATE>

## When to Use

- Validation Pyramid check failed at any layer
- Loss not decreasing / oscillating / diverging
- Training runs but metrics are unexpectedly poor
- GPU utilization or throughput below target
- Any "this doesn't look right" during training

## The Three Core Questions

Every ML diagnostic starts by identifying which question you're answering:

### Q1: Why isn't it converging?
- Loss not decreasing / oscillating / diverging
- Gradient vanishing / exploding
- Activation collapse / attention degradation
- **Start with:** L1 process metrics (gradient, activation, parameter drift)

### Q2: What are these early anomaly signals?
- Loss spike (sudden jump > 10x moving average)
- Abnormal parameter drift from initialization
- MoE entropy anomaly (expert load imbalance)
- Abnormal residual stream write ratio
- **Start with:** L1 architecture-specific checks

### Q3: Why isn't it fast enough?
- I/O bottleneck (data loading starving GPU)
- Low GPU utilization (MFU/TCA below target)
- High communication overhead (NCCL bandwidth insufficient)
- Memory bottleneck (excessive fragmentation, OOM)
- **Start with:** L0 engineering efficiency checks, then per-layer profiling

## The Four Phases

You MUST complete each phase before proceeding to the next.

### Phase 1: Evidence Gathering

**BEFORE proposing ANY fix:**

1. **Identify the core question** (Q1, Q2, or Q3 above)
2. **Collect relevant metrics** using Validation Pyramid checks:
   - Q1/Q2: Run L1 process metrics (gradient norms, activation stats, loss history)
   - Q3: Run L0 engineering checks (MFU, memory, I/O timing)
3. **Record baseline numbers** — you need these to know if your fix helped
4. **Check the obvious first:**
   - Is the learning rate reasonable? (not 0, not 1e10)
   - Is the loss function correct for the task?
   - Is the data pipeline feeding correct labels?
   - Are there NaN/Inf in gradients or activations?

```python
# Quick diagnostic snapshot
def diagnostic_snapshot(model, loss_history, step):
    snapshot = {}

    # Gradient health
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            snapshot[f"grad/{name}/norm"] = g.norm().item()
            snapshot[f"grad/{name}/has_nan"] = torch.isnan(g).any().item()
            snapshot[f"grad/{name}/has_inf"] = torch.isinf(g).any().item()

    # Loss trend
    if len(loss_history) >= 10:
        recent = loss_history[-10:]
        snapshot["loss/recent_mean"] = sum(recent) / len(recent)
        snapshot["loss/recent_std"] = (sum((x - snapshot["loss/recent_mean"])**2 for x in recent) / len(recent)) ** 0.5
        snapshot["loss/trend"] = "decreasing" if recent[-1] < recent[0] else "flat_or_increasing"

    # Parameter norms
    for name, param in model.named_parameters():
        snapshot[f"param/{name}/norm"] = param.data.norm().item()

    return snapshot
```

### Phase 2: Pattern Analysis

**Find the pattern:**

1. **Compare against expected behavior:**
   - What should gradient norms look like at this training stage?
   - What loss range is expected for this task/architecture?
   - What MFU is typical for this GPU + model combo?

2. **Localize the problem:**
   - Is it all layers or specific layers?
   - Is it from the start or did it develop over training?
   - Is it consistent or intermittent?

3. **Hierarchical decomposition** (when needed):
   ```
   Overall problem
       -> Which substructure? (attention / FFN / embedding / MoE routing)
       -> Within that substructure, which operation?
       -> Is it forward, backward, or both?
   ```

   Use `toolkit/profiling/layer_profiler.py` for per-layer timing decomposition.
   Use gradient hooks for per-layer gradient analysis.

### Phase 3: Hypothesis Testing

**Scientific method — one variable at a time:**

1. **Form specific hypothesis:** "I think [specific cause] because [evidence from Phase 1-2]"
2. **Design minimal test:** Change ONE thing to test the hypothesis
3. **Run the test and compare to baseline**
4. **Conclusion:**
   - Hypothesis confirmed -> Phase 4 (fix)
   - Hypothesis rejected -> form NEW hypothesis from the new evidence
   - **After 3 failed hypotheses: question the architecture** (see below)

### Phase 4: Fix and Verify

1. **Implement the fix** based on confirmed hypothesis
2. **Re-run the same Validation Pyramid checks** that originally failed
3. **Compare against Phase 1 baseline** — did the numbers improve?
4. **If fix didn't help:** return to Phase 2 with new information

## Common Diagnostic Patterns

### Loss not decreasing

| Evidence | Likely Cause | First Action |
|----------|-------------|--------------|
| All gradients near 0 | Vanishing gradients | Check init, residual connections, gradient clipping |
| Some gradients exploding | Exploding gradients | Lower LR, add gradient clipping |
| Gradients normal but loss flat | LR too low, or loss function wrong | Check LR schedule, verify loss function |
| Loss oscillating wildly | LR too high | Reduce LR by 10x, check batch size |
| Loss goes to NaN | Numerical instability | Check for log(0), division by 0, overflow |
| Loss decreases then plateaus high | Underfitting | Check model capacity, data quality |

### Efficiency problems

| Evidence | Likely Cause | First Action |
|----------|-------------|--------------|
| MFU < 20% | I/O bottleneck or wrong backend | Check data loading speed, verify FA enabled |
| MFU 20-40% | Memory bound or poor overlap | Check memory profiler, verify comm/compute overlap |
| MFU > 40% but slow | Expected — model is memory-bound | This may be normal for your architecture |
| One layer dominates time | Bottleneck layer | Profile that layer specifically, check for inefficient ops |

### Architecture anomalies

| Evidence | Likely Cause | First Action |
|----------|-------------|--------------|
| Attention entropy < 0.1 | Attention collapse | Check attention init, temperature, positional encoding |
| MoE balance < 0.5 | Expert collapse | Check auxiliary loss, router init |
| Residual write ratio > 1.0 | Layer output overwhelming residual | Check layer norm placement, init scale |
| Embedding norms exploding | Embedding learning rate too high | Separate LR for embeddings |

## The 3-Strike Rule

**If 3 hypotheses fail, STOP fixing and question fundamentals:**

- Is the architecture appropriate for this task?
- Is the data quality/quantity sufficient?
- Is the training recipe (LR, batch size, optimizer) reasonable?
- Are we reproducing a known-working setup, or inventing something new?

**Discuss with the human before attempting more fixes.** Three failures usually mean the problem is architectural, not a parameter tweak.

## Integration with Validation Pyramid

```
Validation Pyramid check fails
    -> Identify which check failed (L0/L1/L2/L3)
    -> Map to core question (Q1/Q2/Q3)
    -> Enter Phase 1 with relevant metrics already collected
    -> After fix: re-run the failed Validation Pyramid check
    -> Pass: return to pyramid (continue to next layer)
    -> Fail again: continue diagnosis
```

## Related Skills

- **spml:vp-engineering-efficiency** — L0 checks that trigger Q3 diagnostics
- **spml:vp-process-metrics** — L1 checks that trigger Q1/Q2 diagnostics
- **spml:vp-overfitting-test** — L2 check that triggers Q1 diagnostics
- **spml:systematic-debugging** — For non-ML bugs (integration, pipeline, data format)
