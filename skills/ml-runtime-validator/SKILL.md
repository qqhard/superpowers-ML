---
name: ml-runtime-validator
description: Use when running L1 runtime validation — trains for ~5 minutes collecting performance/health metrics, then verifies the full pipeline (checkpoint, inference, evaluation) in one sequential run
---

# L1: ML Runtime Validation

## Overview

Run a sequential 6-stage pipeline: load data, instantiate model, train for an estimated 5 minutes of training volume (collecting performance and health metrics), then verify checkpoint save/load, inference, and evaluation. One run, one unified report.

**L1 is validation, not the experiment.** The full training run happens later via `spml:training-handoff` → `spml:watchdog`.

**This is a RIGID skill.** Run all 6 stages. Don't skip stages. Don't skip metrics collection.

## When to Use

- After L0 (spml:ml-static-checks) passes
- Invoked by the orchestrator in `spml:ml-subagent-dev`, not by the Implementer directly
- Skip only if explicitly marked "skip L1" in the experiment design doc

## Data Flow Selection

User declares during brainstorming which data flow to use:
- **Real data flow** — when the dataset is meaningful and overfitting test has no reference value
- **Mock overfit data flow** — small dataset with repeated sampling, for verifying model can fit

## The 6-Stage Pipeline

```
Stage 1: Data Loading
  - Load N batches, verify shapes, no NaN/Inf in inputs
  - Verify DataLoader iterator doesn't crash or hang

Stage 2: Model Instantiation
  - Create model with training config
  - Pass one batch through forward pass
  - Verify output shape correct, no errors

Stage 3: Training (~5 min estimated training volume)
  - forward + backward + optimizer.step
  - Collect simultaneously:
    · Performance: MFU, TCA, throughput, memory
    · Health: loss trend, gradient health, parameter drift
    · Architecture-specific checks (see below)
    · Logging output validation (see below)

Stage 4: Checkpoint
  - Save model state_dict and optimizer state_dict
  - Load into fresh model instance
  - Verify all parameters match (torch.allclose), optimizer state restored

Stage 5: Inference
  - Switch to eval mode
  - Run N forward passes
  - Verify no NaN/Inf in outputs, deterministic (if seeds set)

Stage 6: Evaluation
  - Compute evaluation metrics on N batches
  - Verify metrics are finite numbers, metric function doesn't crash
```

## Stage 3: Metrics Collected (one run, simultaneous)

| Category | Metric | Source |
|----------|--------|--------|
| **Performance** | MFU | FlopCounterMode + CUDA Events |
| **Performance** | TCA | DCGM field 1004 |
| **Performance** | Sample/Token throughput | batch_size / step_time |
| **Performance** | Memory usage | peak / allocated / fragmentation |
| **Training health** | Loss trend | Whether loss is decreasing (not absolute value) |
| **Training health** | Gradient health | NaN/Inf, exploding/vanishing detection |
| **Training health** | Parameter drift | Parameters updating, drift rate |

### Architecture-Specific Checks

Load based on model architecture:

| Architecture | Metric | What to check |
|-------------|--------|--------------|
| Transformer | Attention entropy | Not collapsed (all mass on one token) or uniform (no learning) |
| MoE | Expert load balance | Experts receiving roughly equal traffic, aux loss active |
| RecSys | Embedding stability | Embedding norms not diverging, negative sampling working |
| LLM | KV cache growth | Memory growth linear with sequence length, not quadratic |
| ResNet | Residual write ratio | Residual branch contributing meaningful signal |

### Logging Output Validation

Checks that the training code's logging actually produces correct output at runtime. Each check validates three layers: **existence → frequency → value correctness**.

| # | Check | Severity | Validation Method |
|---|-------|----------|-------------------|
| L.1 | Loss file output correctness | **Mandatory** | File exists, non-empty, parseable format; values reasonable (no all-NaN/Inf/zero, trend consistent with gradient behavior) |
| L.2 | Step speed output correctness | **Mandatory** | File exists, non-empty; values match wall clock (step count × reported step time ≈ actual elapsed time) |
| L.3 | Data loading duration correctness | Advisory | Duration record exists; values reasonable (non-zero, non-negative, consistent with actual time window) |
| L.4 | Output frequency reasonableness | Advisory | Actual log entry timestamps have intervals approximately minute-level (complements L0 check 22 which verifies interval-control logic in code) |
| L.5 | Progress bar correctness | Advisory | Progress bar total matches training target — 1 epoch → dataset size; N steps → total = N; T minutes → time-based estimate; advance rate matches actual speed |
| L.6 | Visualization tool output correctness (if enabled) | **Mandatory** | Output directory/API has data; frequency reasonable; values cross-validated against loss/speed files for consistency; skip if not enabled |

## Failure Detection

Two tiers of thresholds:

### Project-Specific Baselines (configurable)

During brainstorming, user defines acceptable minimums for key metrics. Examples:
- Minimum MFU (e.g., "MFU must be >= 30% on H100")
- Maximum step time (e.g., "step time must be < 200ms")
- Minimum throughput (e.g., "> 10k tokens/sec")

These are stored in the experiment design doc and checked during L1. If no baselines are declared, L1 only checks for anomalies.

### Anomaly Detection (always active)

Catches obvious problems regardless of baselines:
- Loss not decreasing, or actively increasing
- Gradient all NaN/Inf
- MFU abnormally low (< 1%)
- Memory fragmentation extreme
- Architecture-specific metrics degenerate
- Logging outputs missing or empty (L.1, L.2 checks)
- Pipeline stage crash or timeout (stages 1-6)

## Training Volume Control

L1 is a **validation step**, not the experiment itself. You MUST limit training to a short run.

**Default:** An estimated 5 minutes of training volume. User can override during brainstorming.

**How to limit — in order of preference:**

1. **Config/CLI override** (preferred) — pass `--max_steps=N` or `--epochs=M` or equivalent flag directly to the training script. Estimate step count to yield ~5 minutes of training. This is the most reliable method.
2. **Create an L1-specific config file** — copy the training config, modify `epochs`/`max_steps`, and pass it to the script. Keep this config file next to the training script for traceability.
3. **Wrapper script** — write a thin wrapper that modifies the config before calling the training script.

**Do NOT monkey-patch** runtime objects (e.g., patching `trainer.max_epochs` after construction). Monkey-patches fail silently when internal APIs change, causing L1 to run the full experiment.

**Verification (mandatory):** Within the first 30 seconds of L1 execution, confirm that the training process reports the expected limited total (e.g., "Training for M epochs" or "Total steps: N" in the log). If the log shows the full experiment total (e.g., 80 epochs), **kill immediately** — the limiting didn't work. This counts as a failure and enters the fix loop.

## Timeout Protection

**Total timeout:** 15 minutes for the entire L1 run (all 6 stages).

**Timeout is a safety net, not the limiting mechanism.** If L1 hits timeout, it likely means training volume estimation was wrong or a stage hung. Treat timeout as a bug to investigate, not an expected path.

**Background execution liveness check:**
When L1 dispatches stages to background execution, the orchestrator MUST monitor:

1. Start a check loop at **30-second intervals**
2. Each check: is the process still running? Has total timeout been exceeded?
3. **Timeout exceeded** → kill the background process → report as timeout failure → enter fix loop (same as any VP failure)
4. **Process completes within timeout** → read output → continue normal L1 analysis

```
Start runtime validation (6 stages)
    -> Background execution with 30s liveness checks
    -> Total timeout: 15 minutes
    -> Normal completion -> check all metrics + stage results
    -> Timeout -> kill process
        -> Analyze hang cause (deadlock, communication block, data loading stuck, stage hung)
        -> Send to Implementer for fix
        -> Counts toward 5-retry limit
```

**Critical:** Do NOT dispatch to background and then wait indefinitely. A hung process with no timeout detection will stall the entire VP flow.

## Toolkit Usage

Reuse existing `toolkit/profiling/` for performance metrics:
- `l0_runner.py` — main entry for performance collection (named after old L0, used by L1)
- `mfu_calculator.py`, `dcgm_profiler.py`, `gap_analyzer.py` — unchanged
- `memory_profiler.py`, `layer_profiler.py` — unchanged

Training health metrics (gradient hooks, loss recording, arch-specific monitors) are written per-project by the Implementer. Skills guide what to collect; toolkit extraction happens only after proven need.

## Hierarchical Decomposition

When a metric fails, decompose to find root cause before fixing:

```
Overall metric not meeting target
    -> Decompose into substructures
    -> Profile each substructure
    -> Locate bottleneck
    -> Drill to operator level if needed
```

Example: MFU low -> per-layer profiling -> attention layer 3x slower than expected -> missing FlashAttention on that layer.

## Output Report (Unified)

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

## Fix Loop

Uses the shared fix loop from `spml:validation-pyramid`:
- Stage/metric fails -> send to Implementer with specific diagnosis and fix instructions
- Implementer fixes -> re-run entire L1 (from stage 1)
- 5 consecutive failures -> pause, notify user
- If fix modifies > 50 lines -> rollback: re-run Spec Review + Code Quality Review + L0 + L1

## What This Catches

Issues found across the 6 stages:
- **Stages 1-2:** Shape errors, path errors, preprocessing bugs, config errors
- **Stage 3:** Performance anomalies (low MFU, gradient NaN, loss not decreasing), numerical instability
- **Stage 4:** Checkpoint save/load dropping optimizer state or custom buffers, serialization bugs
- **Stage 5:** Eval mode changing behavior unexpectedly (dropout, batch norm), output NaN
- **Stage 6:** Metric function bugs, label format errors, data iterator exhaustion

## Integration

- **spml:ml-subagent-dev** — invokes this as a validation stage
- **spml:validation-pyramid** — L1 in the 2-level pyramid
- **spml:ml-static-checks** — must pass before L1 runs
- **spml:diagnostics** — triggered on failure for root cause analysis
