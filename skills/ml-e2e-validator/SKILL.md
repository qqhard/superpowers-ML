---
name: ml-e2e-validator
description: Use when running L2 end-to-end pipeline validation — verifies each pipeline stage runs through with 1-5 steps per stage
---

# L2: ML End-to-End Pipeline Validation

## Overview

Verify the full pipeline runs through, each stage running a small number of steps. Not testing performance or result quality — testing that the flow is correct. If the flow is broken, it surfaces here before wasting hours on full training.

**This is a RIGID skill.** Run all 6 stages. Don't skip stages.

## When to Use

- After L1 (spml:ml-runtime-validator) passes
- Invoked by the orchestrator in `spml:ml-subagent-dev`, not by the Implementer directly
- Skip only if explicitly marked "skip L2" in the experiment design doc

## 6 Stages

Default: 1 step per stage. Configurable to 3-5 steps per stage (declared during brainstorming). Running multiple steps has minimal extra cost but significantly improves coverage — catches issues like shape mismatches on the second batch, accumulation bugs across steps, or non-deterministic failures.

| Stage | Validates | Typical issues exposed |
|-------|-----------|----------------------|
| 1. Data loading | N batches load correctly | Shape errors, path errors, preprocessing bugs, iterator exhaustion |
| 2. Model instantiation | Model creates, accepts input | Config errors, layer definition bugs |
| 3. Training N steps | fwd + bwd + optimizer.step x N | Gradient errors, shape mismatch, accumulation bugs |
| 4. Checkpoint save/load | save -> load -> params match | Serialization bugs, incomplete state_dict |
| 5. Inference | eval mode, N steps | dropout/BN behavior, output NaN |
| 6. Evaluation | metric computation on N batches | Metric function bugs, label format errors |

### Stage Details

**Stage 1: Data Loading**
- Load N batches from the configured data source
- Verify: shapes match model expectations, dtypes correct, no NaN/Inf in input
- Verify: DataLoader iterator doesn't crash or hang

**Stage 2: Model Instantiation**
- Create model with training config
- Pass one batch through forward pass
- Verify: output shape correct, no errors

**Stage 3: Training N Steps**
- Run N complete training steps (forward + backward + optimizer.step)
- Verify: no errors, loss is a finite scalar, gradients exist

**Stage 4: Checkpoint Save/Load**
- Save model state_dict and optimizer state_dict
- Load into a fresh model instance
- Verify: all parameters match (torch.allclose), optimizer state restored

**Stage 5: Inference**
- Switch to eval mode
- Run N forward passes
- Verify: no NaN/Inf in outputs, outputs are deterministic (if seeds set)

**Stage 6: Evaluation**
- Compute evaluation metrics on N batches
- Verify: metrics are finite numbers, metric function doesn't crash

## Timeout Protection

**Per-stage timeout:** 2 minutes (configurable). Single stage hanging beyond timeout is killed and counts as a failure.

**Overall timeout:** 15 minutes for the entire L2 run.

**Background execution liveness check:**
When L2 dispatches pipeline stages to background execution, the orchestrator MUST monitor them:

1. Start a check loop at **30-second intervals**
2. Each check: is the process still running? Has per-stage or overall timeout been exceeded?
3. **Timeout exceeded** → kill the background process → report which stage timed out → enter fix loop (same as any VP failure)
4. **Process completes within timeout** → read output → continue to next stage

**Critical:** Do NOT dispatch to background and then wait indefinitely. A hung process with no timeout detection will stall the entire VP flow.

## Difference from L1

| | L1 Runtime Validation | L2 End-to-End |
|--|----------------------|---------------|
| **Purpose** | Performance + training health | Flow correctness |
| **Duration** | Several minutes continuous | N steps per stage, completes quickly |
| **Cares about** | Speed, utilization, loss trend, metric anomalies | Does each stage run without error |
| **Does NOT care about** | Final result quality | How fast it runs |

## Fix Loop

Uses the shared fix loop from `spml:validation-pyramid`:
- Stage fails -> send to Implementer with stage number, error message, and context
- Implementer fixes -> re-run all stages from stage 1 (don't skip)
- 5 consecutive failures -> pause, notify user
- If fix modifies > 50 lines -> rollback: re-run Spec Review + Code Quality Review + L0 + L1 + L2

## What This Catches

Common issues found at L2 that earlier levels miss:
- Shape mismatches between model output and loss function input
- Checkpoint save/load dropping optimizer state or custom buffers
- Eval mode changing behavior unexpectedly (dropout, batch norm)
- Metric function expecting different label format than training produces
- Data iterator exhausting and not resetting properly
- Device mismatches that only appear in the full pipeline (not individual components)

## Integration

- **spml:ml-subagent-dev** — invokes this as a validation stage
- **spml:validation-pyramid** — L2 in the 3-level pyramid
- **spml:ml-runtime-validator** — must pass before L2 runs
- **spml:diagnostics** — triggered on failure for root cause analysis
