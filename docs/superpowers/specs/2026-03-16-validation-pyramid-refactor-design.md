# Validation Pyramid Refactor Design

## Problem

The current Validation Pyramid (VP) has 4 layers (L0-L3) with 20+ skill files. It's overly complex, and its post-hoc "pass/fail" reporting model means problems surface too late — after implementation is done. The VP runs as a separate phase disconnected from the code-writing process, so errors accumulate until a user finally sees a failure report.

## Design Goals

1. Simplify from 4 layers to 3 levels with clear purposes
2. Integrate into Superpowers' existing subagent-driven-development flow — not a separate system
3. Shift from "report problems after the fact" to "catch and fix during implementation"
4. Shared fix loop: detect → auto-fix → retry → 3 failures → user intervention

## Architecture Overview

The new VP adds 3 ML validation stages after Superpowers' existing review pipeline:

```
Subagent-Driven-Development (per task)
├─ Implementer writes code
├─ Spec Reviewer (unchanged)
├─ Code Quality Reviewer (unchanged)
├─ L0: ML Code Reviewer (spml:ml-code-reviewer)       — static analysis
├─ L1: ML Runtime Validator (spml:ml-runtime-validator) — minutes-level run
├─ L2: ML E2E Validator (spml:ml-e2e-validator)        — 1-step-per-stage pipeline
└─ All pass → task complete
```

Execution order: L0 must pass before L1. L1 must pass before L2. Each level uses the same fix loop.

### Shared Fix Loop

All three levels share one mechanism:

```
Run validation
    → Pass → proceed to next level
    → Fail → send feedback to Implementer with specific issues
        → Implementer fixes → re-run validation
        → 3 consecutive failures → pause, notify user for intervention
```

Timeout counts as a failure. Timeouts and metric failures share the same retry counter.

## L0: Static Analysis (ML Code Reviewer)

**What:** A ML-specialized code-reviewer subagent with a conditional checklist. Each rule has an applicability condition — only checked when the condition is met.

**When:** After Spec Review and Code Quality Review pass.

**How:** Fork Superpowers' `code-reviewer.md` agent definition into SPML, register as `spml:ml-code-reviewer`. Add the ML checklist. The reviewer reads code and checks applicable rules — if a rule fails, it gives specific fix instructions (file:line) and sends the Implementer back to fix. Uses the same Critical/Important/Minor severity levels as Superpowers.

### Checklist

| # | Check | Condition | What to verify |
|---|-------|-----------|---------------|
| 1 | Device consistency | Uses CUDA | Model, data, loss on same device |
| 2 | Precision config | Has mixed precision / bf16 / fp16 | param.dtype matches expectation, autocast correct |
| 3 | FlashAttention | Has Attention layers | FA available and enabled |
| 4 | Optimizer coverage | Always | optimizer param_groups covers all trainable params |
| 5 | LR scheduler | Has lr_scheduler | Correctly linked to optimizer |
| 6 | DataLoader config | Has DataLoader | num_workers, pin_memory reasonable |
| 7 | Data loading method | Large dataset | Uses mmap loading |
| 8 | Padding waste | Variable-length sequences | Excessive padding or tail-wave waste |
| 9 | Random seeds | Always | torch/np/random seeds set |
| 10 | Gradient accumulation consistency | Has gradient accumulation | accumulation_steps × micro_batch = global_batch |
| 11 | Loss reduction | Has gradient accumulation | mean vs sum matches accumulation strategy |
| 12 | Vocab/Embedding match | Has Embedding layer | tokenizer vocab_size == embedding dim |
| 13 | Frozen layers | Fine-tuning | Frozen layers match expectations |
| 14 | FLOPs estimate | Always | FlopCounterMode static count, report theoretical compute |
| 15 | GPU hardware info | Uses CUDA | Model, peak TFLOPS, memory capacity |
| 16 | Memory estimate | Always | Param count, theoretical memory fits GPU capacity |
| 17 | MoE backend | MoE architecture | Expert parallel, routing optimization, aux loss |
| 18 | CUDA kernel selection | Uses CUDA | Optimized kernels, not fallback |

This checklist is designed for ongoing maintenance — items can be added as new common mistakes are discovered.

## L1: Runtime Validation (ML Runtime Validator)

**What:** Actually run training for a few minutes, collecting performance and training health metrics simultaneously.

**When:** After L0 passes.

### Data Flow Selection

User declares during brainstorming which data flow to use:
- **Real data flow** — when the dataset is meaningful and overfitting test has no reference value
- **Mock overfit data flow** — small dataset with repeated sampling, for verifying model can fit

### Metrics Collected (one run, simultaneous)

| Category | Metric | Source |
|----------|--------|--------|
| **Performance** | MFU | FlopCounterMode + CUDA Events |
| **Performance** | TCA | DCGM field 1004 |
| **Performance** | Sample/Token throughput | batch_size / step_time |
| **Performance** | Memory usage | peak / allocated / fragmentation |
| **Training health** | Loss trend | Whether loss is decreasing (not absolute value) |
| **Training health** | Gradient health | NaN/Inf, exploding/vanishing detection |
| **Training health** | Parameter drift | Parameters updating, drift rate |
| **Arch-specific** | Attention entropy | Transformer |
| **Arch-specific** | Expert load balance | MoE |
| **Arch-specific** | Embedding stability | RecSys |
| **Arch-specific** | KV cache growth | LLM |
| **Arch-specific** | Residual write ratio | ResNet |

### Failure Detection

No absolute thresholds. Use obvious anomaly detection:
- Loss not decreasing or increasing
- Gradient all NaN/Inf
- MFU abnormally low (< 1%)
- Memory fragmentation extreme
- Architecture-specific metrics degenerate

### Timeout Protection

```
Start runtime validation (expected N minutes)
    → Timeout = N × 1.5 (e.g., expect 5 min → timeout 7.5 min)
    → Normal completion → check metrics
    → Timeout → kill process
        → Analyze hang cause (deadlock, communication block, data loading stuck)
        → Send to Implementer for fix
        → Counts toward 3-retry limit
```

### Toolkit Usage

Reuse existing `toolkit/profiling/` for performance metrics:
- `l0_runner.py` — main entry for performance collection
- `mfu_calculator.py`, `dcgm_profiler.py`, `gap_analyzer.py` — unchanged
- `memory_profiler.py`, `layer_profiler.py` — unchanged

Training health metrics (gradient hooks, loss recording, arch-specific monitors) are NOT extracted into toolkit yet. Skills guide the Agent to write per-project monitoring code. Extract into toolkit only after practice shows Agent consistently gets them wrong.

## L2: End-to-End Validation (ML E2E Validator)

**What:** Verify the full pipeline runs through, each stage 1 step only. Not testing performance or quality — testing that the flow is correct.

**When:** After L1 passes.

### 6 Stages (1 step each)

| Stage | Validates | Typical issues exposed |
|-------|-----------|----------------------|
| 1. Data loading | One batch loads correctly | Shape errors, path errors, preprocessing bugs |
| 2. Model instantiation | Model creates, accepts input | Config errors, layer definition bugs |
| 3. Training 1 step | fwd + bwd + optimizer.step | Gradient errors, shape mismatch |
| 4. Checkpoint save/load | save → load → params match | Serialization bugs, incomplete state_dict |
| 5. Inference | eval mode, 1 step | dropout/BN behavior, output NaN |
| 6. Evaluation | metric computation on 1 batch | Metric function bugs, label format errors |

### Timeout Protection

Same as L1 — each stage has a timeout. Single stage hanging beyond timeout is killed and counts as a failure.

### Difference from L1

| | L1 Runtime Validation | L2 End-to-End |
|--|----------------------|---------------|
| **Purpose** | Performance + training health | Flow correctness |
| **Duration** | Several minutes continuous | 1 step per stage, completes quickly |
| **Cares about** | Speed, utilization, loss trend, metric anomalies | Does each stage run without error |
| **Does NOT care about** | Final result quality | How fast it runs |

## File Plan

### New files

```
skills/
├─ validation-pyramid/
│   └─ SKILL.md                    ← rewrite: 3-level definition + shared fix loop
├─ ml-code-reviewer/               ← new (L0)
│   ├─ SKILL.md                    ← trigger timing, integration description
│   └─ checklist.md                ← conditional checklist (maintainable)
├─ ml-runtime-validator/           ← new (L1)
│   └─ SKILL.md                    ← data flow selection, metrics, timeout
├─ ml-e2e-validator/               ← new (L2)
│   └─ SKILL.md                    ← 6 stages, timeout
├─ subagent-dev/
│   └─ SKILL.md                    ← modify: chain L0→L1→L2 after code review

agents/
├─ ml-code-reviewer.md             ← fork from Superpowers code-reviewer.md + ML checklist
```

### Delete (old VP files)

```
skills/
├─ vp-engineering-efficiency/      ← delete (merged into L0 + L1)
├─ vp-process-metrics/             ← delete (merged into L1)
├─ vp-overfitting-test/            ← delete (no longer exists as separate level)
├─ vp-e2e-pipeline/                ← delete (rewritten as ml-e2e-validator)
```

### Unchanged

```
toolkit/profiling/                 ← no changes, L1 reuses as-is
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Number of levels | 3 (was 4) | Simpler; overfitting test absorbed into L1 data flow choice |
| Integration point | After existing Superpowers review stages | Reuse infrastructure, consistent UX |
| Fix mechanism | Auto-fix loop with 3-retry cap | Agent fixes most issues; user only involved for hard problems |
| Timeout handling | Kill + count as failure | Prevents hanging tests from blocking the pipeline |
| Toolkit extraction | Only for proven-difficult code | Avoid premature abstraction |
| Superpowers changes | Fork into SPML, spml: prefix | Minimize cross-project coupling |
| Checklist design | Conditional rules | Not every check applies to every project |
