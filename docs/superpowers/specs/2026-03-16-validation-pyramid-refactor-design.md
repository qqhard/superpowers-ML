# Validation Pyramid Refactor Design

## Problem

The current Validation Pyramid (VP) has 4 layers (L0-L3) with 20+ skill files. It's overly complex, and its post-hoc "pass/fail" reporting model means problems surface too late — after implementation is done. The VP runs as a separate phase disconnected from the code-writing process, so errors accumulate until a user finally sees a failure report.

## Design Goals

1. Simplify from 4 layers to 3 levels with clear purposes
2. Integrate into Superpowers' existing subagent-driven-development flow — not a separate system
3. Shift from "report problems after the fact" to "catch and fix during implementation"
4. Shared fix loop: detect → auto-fix → retry → 5 failures → user intervention

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

**Behavioral change from current design:** The current `subagent-dev` has the Implementer running VP *during* implementation. The new design moves ML validation to *after* code reviews, as separate orchestrator-dispatched stages. This ensures code quality is verified before spending GPU time on runtime validation.

### Subagent Dispatch Model

L0 runs as a **subagent** (the ml-code-reviewer agent, dispatched by the orchestrator like spec-reviewer and code-quality-reviewer). L1 and L2 run as **skills invoked by the orchestrator** — the orchestrator runs the validation commands directly, since these are execution tasks, not review tasks.

When any level fails, the orchestrator sends the **same Implementer subagent** back to fix the issue (resumed, not a new instance). After the Implementer fixes, re-run only the failed level — do not re-run earlier reviews (Spec Review, Code Quality Review, or earlier VP levels that already passed).

### Shared Fix Loop

All three levels share one mechanism. Each level has its own retry counter (resets when advancing to the next level):

```
Run validation
    → Pass → proceed to next level (reset counter)
    → Fail → send feedback to Implementer with specific issues
        → Implementer fixes → re-run this level
        → 5 consecutive failures at this level → pause, notify user
```

Timeout counts as a failure. Timeouts and metric failures share the same per-level retry counter.

### Large Fix Rollback Rule

If the Implementer's fix modifies more than 50 lines of code, the fix is considered a substantial change. In this case, **roll back and re-run all previous stages**: Spec Review → Code Quality Review → L0 (and any passed levels before the current one). This prevents large fixes from introducing new problems that earlier reviews would have caught.

## L0: Static Analysis (ML Code Reviewer)

**What:** A ML-specialized code-reviewer subagent with a conditional checklist. Each rule has an applicability condition — only checked when the condition is met.

**When:** After Spec Review and Code Quality Review pass.

**How:** Copy Superpowers' `agents/code-reviewer.md` into SPML's `agents/ml-code-reviewer.md` and modify it to include the ML checklist. Register as a named agent with frontmatter `name: ml-code-reviewer` — SPML skills reference it as `spml:ml-code-reviewer` via the `spml:` skill prefix. The reviewer reads code and checks applicable rules — if a rule fails, it gives specific fix instructions (file:line) and sends the Implementer back to fix. Uses the same Critical/Important/Minor severity levels as Superpowers.

### Checklist

The checklist is split into two severity tiers:

**Mandatory (Critical) — checks 1-6.** Failure blocks progress; Implementer must fix before proceeding.

| # | Check | Condition | What to verify |
|---|-------|-----------|---------------|
| 1 | Device consistency | Uses CUDA | Model, data, loss on same device |
| 2 | Precision config | Has mixed precision / bf16 / fp16 | param.dtype matches expectation, autocast correct |
| 3 | FlashAttention | Has Attention layers | FA available and enabled |
| 4 | Optimizer coverage | Always | optimizer param_groups covers all trainable params |
| 5 | LR scheduler | Has lr_scheduler | Correctly linked to optimizer |
| 6 | DataLoader config | Has DataLoader | num_workers, pin_memory reasonable |

**Advisory (Warning) — checks 7-18.** Do not block progress. Reported as warnings; Implementer may fix or acknowledge and proceed.

| # | Check | Condition | What to verify |
|---|-------|-----------|---------------|
| 7 | Data loading method | Dataset declared as large in brainstorming, or file size > 10GB | Uses mmap loading |
| 8 | Padding waste | Variable-length sequences | Excessive padding or tail-wave waste |
| 9 | Random seeds | Always | torch/np/random seeds set |
| 10 | Gradient accumulation consistency | Has gradient accumulation | accumulation_steps × micro_batch = global_batch |
| 11 | Loss reduction | Has gradient accumulation | mean vs sum matches accumulation strategy |
| 12 | Vocab/Embedding match | Has Embedding layer | tokenizer vocab_size == embedding dim |
| 13 | Frozen layers | Fine-tuning | Frozen layers match expectations |
| 14 | FLOPs estimate | Always | Estimate from model architecture (param dims, known FLOPs-per-op), order-of-magnitude check. NOT FlopCounterMode — that requires runtime, used in L1 |
| 15 | GPU hardware info | Uses CUDA | Review code for target GPU assumptions; actual hardware detection happens in L1 |
| 16 | Memory estimate | Always | Estimate from param count + dtype + expected activations; check if theoretical footprint fits target GPU capacity |
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

Two tiers of thresholds:

**Project-specific baselines (configurable):** During brainstorming, user defines acceptable minimums for key metrics. Examples:
- Minimum MFU (e.g., "MFU must be ≥ 30% on H100")
- Maximum step time (e.g., "step time must be < 200ms")
- Minimum throughput (e.g., "> 10k tokens/sec")

These are stored in the experiment design doc and checked during L1. If no baselines are declared, L1 only checks for anomalies (see below).

**Anomaly detection (always active):** Catches obvious problems regardless of baselines:
- Loss not decreasing, or actively increasing
- Gradient all NaN/Inf
- MFU abnormally low (< 1%)
- Memory fragmentation extreme
- Architecture-specific metrics degenerate

### Timeout Protection

L1 default runtime: 5 minutes. User can override during brainstorming. Timeout = configured runtime × 1.5.

```
Start runtime validation (default 5 min, configurable)
    → Timeout = runtime × 1.5 (e.g., 5 min → timeout 7.5 min)
    → Normal completion → check metrics
    → Timeout → kill process
        → Analyze hang cause (deadlock, communication block, data loading stuck)
        → Send to Implementer for fix
        → Counts toward 5-retry limit
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

### 6 Stages

Default: 1 step per stage. Configurable to 3-5 steps per stage (declared during brainstorming). Running multiple steps has minimal extra cost but significantly improves coverage — catches issues like shape mismatches on the second batch, accumulation bugs across steps, or non-deterministic failures.

| Stage | Validates | Typical issues exposed |
|-------|-----------|----------------------|
| 1. Data loading | N batches load correctly | Shape errors, path errors, preprocessing bugs, iterator exhaustion |
| 2. Model instantiation | Model creates, accepts input | Config errors, layer definition bugs |
| 3. Training N steps | fwd + bwd + optimizer.step × N | Gradient errors, shape mismatch, accumulation bugs |
| 4. Checkpoint save/load | save → load → params match | Serialization bugs, incomplete state_dict |
| 5. Inference | eval mode, N steps | dropout/BN behavior, output NaN |
| 6. Evaluation | metric computation on N batches | Metric function bugs, label format errors |

### Timeout Protection

Each stage has a default timeout of 2 minutes (configurable). Single stage hanging beyond timeout is killed and counts as a failure. The entire L2 run has an overall timeout of 10 minutes.

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

Note: `toolkit/profiling/l0_runner.py` is named after the old L0 layer but is now used by the new L1 for runtime performance collection. This naming mismatch is acknowledged and acceptable — renaming would break existing tests and imports for no functional benefit.

### Migration: Update References to Old VP Skills

The following files reference old VP skill names and must be updated during implementation:

- `skills/brainstorming/SKILL.md` — references to VP scope selection
- `skills/diagnostics/SKILL.md` — references to `spml:vp-*` skills
- `skills/experiment-planning/SKILL.md` — references to VP layer names
- `skills/subagent-dev/SKILL.md` — VP integration in workflow
- `skills/validation-pyramid/SKILL.md` — complete rewrite
- Any other skills referencing `vp-engineering-efficiency`, `vp-process-metrics`, `vp-overfitting-test`, `vp-e2e-pipeline`

### Content from Current `validation-pyramid/SKILL.md`

The current SKILL.md contains: TDD RED-GREEN-REFACTOR rhythm, orchestration logic, hierarchical decomposition, three granularity levels, and red flags. In the rewrite:

- **TDD rhythm** — absorbed into the fix loop. The fix loop IS the RED-GREEN cycle: validation fails (RED), implementer fixes (GREEN). Explicit TDD framing is dropped since the loop enforces it mechanically.
- **Orchestration logic** — replaced by the L0→L1→L2 chain in `subagent-dev/SKILL.md`.
- **Hierarchical decomposition** — retained in L1 (when metrics fail, decompose to find root cause before fixing).
- **Three granularity levels** — dropped. The new 3-level design replaces this.
- **Red flags** — retained and updated for the new 3-level design.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Number of levels | 3 (was 4) | Simpler; overfitting test absorbed into L1 data flow choice |
| Integration point | After existing Superpowers review stages | Reuse infrastructure, consistent UX |
| Fix mechanism | Auto-fix loop with 5-retry cap | Agent fixes most issues; user only involved for hard problems |
| Timeout handling | Kill + count as failure | Prevents hanging tests from blocking the pipeline |
| Toolkit extraction | Only for proven-difficult code | Avoid premature abstraction |
| Superpowers changes | Fork into SPML, spml: prefix | Minimize cross-project coupling |
| Checklist design | Conditional rules | Not every check applies to every project |
| VP timing | After code reviews, not during implementation | Code quality verified before spending GPU time |
| Retry counter scope | Per-level (5 max), resets on advance | Prevents one flaky level from exhausting retries for later levels |
| l0_runner.py naming | Keep old name despite L0→L1 shift | Renaming breaks tests/imports for no functional benefit |
| L0 checklist severity | Checks 1-6 mandatory, 7-18 advisory | Core config errors block; optimization hints don't block |
| L1 baselines | Project-specific configurable baselines | Users define "acceptable MFU" etc. during brainstorming |
| L2 step count | Default 1, configurable 3-5 | Minimal extra cost, significantly better coverage |
| Large fix rollback | Fixes > 50 lines re-run all prior reviews + L0 | Prevents large changes from introducing new problems |
