---
name: ml-runtime-validator
description: Use when running L1 runtime validation — performance metrics, training health, and timeout protection during minutes-level training run
---

# L1: ML Runtime Validation

## Overview

Run training for a few minutes, collecting performance and training health metrics simultaneously. Catches issues that static analysis cannot detect: slow performance, numerical instability, gradient problems, and architecture-specific anomalies.

**This is a RIGID skill.** Run all applicable checks. Don't skip metrics collection.

## When to Use

- After L0 (spml:ml-static-checks) passes
- Invoked by the orchestrator in `spml:subagent-dev`, not by the Implementer directly
- Skip only if explicitly marked "skip L1" in the experiment design doc

## Data Flow Selection

User declares during brainstorming which data flow to use:
- **Real data flow** — when the dataset is meaningful and overfitting test has no reference value
- **Mock overfit data flow** — small dataset with repeated sampling, for verifying model can fit

## Metrics Collected (one run, simultaneous)

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

## Timeout Protection

L1 default runtime: 5 minutes. User can override during brainstorming. Timeout = configured runtime x 1.5.

```
Start runtime validation (default 5 min, configurable)
    -> Timeout = runtime x 1.5 (e.g., 5 min -> timeout 7.5 min)
    -> Normal completion -> check metrics
    -> Timeout -> kill process
        -> Analyze hang cause (deadlock, communication block, data loading stuck)
        -> Send to Implementer for fix
        -> Counts toward 5-retry limit
```

## Toolkit Usage

Reuse existing `toolkit/profiling/` for performance metrics:
- `l0_runner.py` — main entry for performance collection (named after old L0, used by new L1)
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

## Fix Loop

Uses the shared fix loop from `spml:validation-pyramid`:
- Metric fails -> send to Implementer with specific diagnosis and fix instructions
- Implementer fixes -> re-run L1
- 5 consecutive failures -> pause, notify user
- If fix modifies > 50 lines -> rollback: re-run Spec Review + Code Quality Review + L0 + L1

## Integration

- **spml:subagent-dev** — invokes this as a validation stage
- **spml:validation-pyramid** — L1 in the 3-level pyramid
- **spml:ml-static-checks** — must pass before L1 runs
- **spml:ml-e2e-validator** — next level after L1 passes
- **spml:diagnostics** — triggered on failure for root cause analysis
