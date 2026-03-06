# Superpowers-ML Design Document

> **For Claude:** This is a design document for a new project. Use superpowers:writing-plans to create the implementation plan.

**Project:** superpowers-ml

**Source:** Fork from [obra/superpowers](https://github.com/obra/superpowers), independent development

**Date:** 2026-03-06

---

## 1. Project Positioning

**Name:** superpowers-ml

**Positioning:** ML/RecSys/LLM training development workflow framework for AI agents. Fork from Superpowers, restructuring all core workflows for ML development.

**Target users:** AI coding agents (Claude Code, Codex, OpenCode, Cursor), serving ML engineering teams, later expanding to community.

**Supported platforms:** Maintain existing Claude Code / Codex / OpenCode / Cursor support.

**Core principles:**

1. In traditional software, code runs = result correct. In ML, code runs ≠ result correct.
2. **"Not working" is reasonable in ML, but the process must be correct.** If an implementation error causes poor results, you may misjudge the experimental strategy itself as ineffective, wasting an entire research direction.
3. Replace traditional TDD with a **Validation Pyramid** — multi-layer process metrics to separate "implementation error" from "strategy ineffective."
4. Retain function/operator-level traditional testing for deterministic code; the Validation Pyramid manages the training process.
5. Users can skip any validation layer — scope confirmed during brainstorm phase.
6. Validation Pyramid dynamically orchestrates based on architecture type, task type, and user context.
7. Only codify toolkit code that agents struggle to write correctly from scratch and is highly reusable; everything else is guided by skills for agents to write on the spot.

---

## 2. Project Structure

```
superpowers-ml/
  # === Core Skills ===
  skills/
    ml-brainstorming/              # Experiment design + context collection + validation scope confirmation
    ml-experiment-planning/        # Experiment decomposition: shared infra + atomic subtasks
    validation-pyramid/            # Layered validation orchestration (dynamic routing)
      SKILL.md
      decision-tree.md             # Architecture -> check item selection logic
      layer-overview.md            # Layer quick reference

    # Validation Pyramid layers (sub-files loaded on demand)
    vp-engineering-efficiency/
      SKILL.md
      backend-checks.md            # FA/MoE/NCCL/HBM/PCIE
      gpu-utilization.md           # MFU/TCA/memory
      distributed-training.md      # Multi-node specific
    vp-numerical-health/
      SKILL.md
      gradient-checks.md
      attention-checks.md          # Transformer specific
      moe-checks.md                # MoE specific
      residual-stream.md           # Residual connection specific
      parameter-drift.md           # Parameter drift, loss spike
    vp-overfitting-test/
      SKILL.md
    vp-domain-metrics/
      SKILL.md
      recsys-metrics.md
      llm-metrics.md
    vp-e2e-pipeline/
      SKILL.md

    # Restructured workflow skills
    ml-diagnostics/                # Non-convergence / early anomaly / efficiency bottleneck
    ml-subagent-dev/               # ML-adapted subagent execution + review
    ml-verification/               # Completion criteria = pyramid passed + valid conclusions

    # Pluggable framework knowledge (async loaded)
    frameworks/
      pytorch/
      huggingface/
      megatron/
      deepspeed/
      sglang/
      vllm/
      wandb/

    # Reused skills (minor adjustments or direct reuse)
    using-superpowers-ml/
    writing-skills/
    dispatching-parallel-agents/
    finishing-a-development-branch/
    receiving-code-review/
    requesting-code-review/

  # === Toolkit (reusable Python tools) ===
  toolkit/
    profiling/
      mfu_calculator.py            # Theoretical FLOPS + measurement + MFU/TCA
      layer_profiler.py            # Per-layer forward/backward timing breakdown
      memory_profiler.py           # Memory analysis
    monitors/
      gradient_monitor.py          # Gradient distribution hooks
      activation_monitor.py        # Activation statistics hooks
      loss_tracker.py              # Loss spike + trend analysis
      parameter_drift.py           # Parameter drift tracking

  # === Multi-platform support ===
  hooks/
  .claude-plugin/
  .cursor-plugin/
  .codex/
  .opencode/
  lib/
  commands/                        # ml-brainstorm, ml-plan, ml-execute
  agents/
  docs/plans/
  tests/
```

---

## 3. Core Workflow

### Full pipeline

```
ml-brainstorming
    |
    Output: experiment design + validation config table
    |
ml-experiment-planning
    |
    Output: shared scaffold definition + atomic subtask list
    |
ml-subagent-dev (execute subtasks one by one)
    |
    Each subtask:
    +-- 1. Function/operator-level unit test (deterministic code)
    +-- 2. Implementation code
    +-- 3. validation-pyramid (dynamically orchestrated per config)
    |       L0 -> L1 -> L2 -> L3 -> L4
    |       Any layer fails -> ml-diagnostics
    |       Diagnostics support hierarchical decomposition (whole -> substructure -> operator)
    +-- 4. Spec review (does implementation match experiment design?)
    +-- 5. Record conclusion (metric data + effective/ineffective/needs further study)
    |
ml-verification (all subtasks complete)
    |
    Output: conclusion summary + recommendations
    |
Human decides: finish / add subtasks / new brainstorm round
```

### 3.1 ml-brainstorming

**Adapted from:** brainstorming

**Retained:** One question at a time, multiple choice preferred, propose 2-3 approaches, confirm design section by section.

**New: Task type identification (first step)**

| Task Type | Validation Approach |
|-----------|-------------------|
| Experiment / ablation | Full Validation Pyramid (L0-L4) |
| Dataset preparation | Data quality checks (distribution, missing values, leakage detection, train/val/test consistency) |
| Pipeline setup | Engineering efficiency (L0) + end-to-end pass (L4) |
| Reproduce baseline | Align with paper metrics + Validation Pyramid confirms correct implementation |
| Inference / deployment | Inference efficiency (latency, throughput, memory) + accuracy alignment |
| Pure engineering optimization | Before/after performance comparison + correctness unchanged |

**New: Experiment design**
- Hypothesis: Doing X is expected to cause Y
- Independent variable: What changes in this experiment
- Dependent variable: What metrics to observe
- Control variable: What stays the same

**New: Context collection**
- Model architecture (Transformer / MoE / CNN / other)
- Task type (RecSys / LLM / CV / other)
- Scale (single GPU / multi-GPU / multi-node)
- Existing infra (data pipeline / training loop / checkpoint / evaluation)

**New: Validation scope confirmation**
- Ask layer by layer: needed / skip / covered by existing infra
- Auto-recommend applicable sub-checks based on architecture type
- User can override any recommendation

**New: Model structure decomposition confirmation**
- Set reasonable segmentation granularity
- Which custom functions/operators need unit tests

**Output:** Design document + validation config table (YAML)

Validation config table example:

```yaml
task_type: experiment
architecture: transformer-moe
task_domain: recsys
distributed: multi-node
user_infra:
  data_pipeline: existing    # Don't touch, advise if problems found
  training_loop: existing
  checkpoint: existing

validation_scope:
  L0_engineering:
    backend_checks: true
    moe_backend: true
    nccl_bandwidth: true
    io_speed: skip           # User has existing infra
  L1_numerical:
    gradient: true
    attention_checks: true
    moe_entropy: true
    residual_stream: true
  L2_overfit: true
  L3_domain:
    recsys_metrics: true
    llm_metrics: skip
  L4_e2e: true

unit_tests:
  data_preprocessing: true
  custom_loss: true
  custom_layers: [MoERouter, SparseAttention]

structure_decomposition:
  segments:
    - name: attention_block
      layers: [MultiHeadAttention, LayerNorm, Residual]
    - name: ffn_block
      layers: [FFN, LayerNorm, Residual]
    - name: moe_routing
      layers: [MoERouter, ExpertFFN]
  mock_data: true
  check_per_segment: [forward_time, backward_time, memory]
```

### 3.2 ml-experiment-planning

**Adapted from:** writing-plans

**Retained:** Bite-sized steps, exact file paths, complete code, executable commands.

**Output structure:**

```markdown
# [Experiment Name] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-ml:ml-subagent-dev

**Goal:** [one sentence]
**Hypothesis:** [doing X is expected to cause Y]
**Validation config:** [reference brainstorm output YAML]

---

## Shared Scaffold

### Existing infra (don't touch, advise if problems found)
- Data pipeline: `path/to/data_loader.py`
- Training loop: `path/to/trainer.py`

### Needs setup
- Evaluation framework: [Validation Pyramid checker integration]
- Model structure decomposition: [segment definitions]

---

## Subtask 1: [experiment name]

**Hypothesis:** [specific hypothesis]
**Implementation:** [what files to change, what logic to change]
**Unit Test:** [which custom functions need traditional tests]
**Validation Pyramid:** [applicable layers + specific metrics + thresholds]
**Expected Conclusion:** [what success means / what failure means]

## Subtask 2: ...
```

### 3.3 validation-pyramid

**Replaces:** test-driven-development

**Orchestration logic:** Read validation config table, execute enabled checks in L0->L4 order, pass each layer before entering next, failure triggers ml-diagnostics.

**Three granularity levels:**

| Granularity | Applicable Scenario | Method |
|-------------|-------------------|--------|
| Function/operator | Custom loss, custom layer, single operator efficiency | Traditional unit test, deterministic assertions |
| Module/layer | Network substructure efficiency, segmented validation | Mock data, segments defined in brainstorm |
| Experiment | L0-L4 full pyramid | Training process metrics |

**Dynamic selection:**
- Transformer -> load attention-checks, residual-stream
- MoE -> load moe-checks
- Multi-node -> load distributed-training
- User-skipped layers -> don't execute

**Hierarchical decomposition on failure:**

```
Overall not meeting target
    -> Decompose into substructures
    -> Validate each with mock data
    -> Locate bottleneck
    -> Drill to operator level if needed
```

### 3.4 Validation Pyramid Layer Details

#### L0: Engineering Efficiency

| Category | Metrics |
|----------|---------|
| I/O & bandwidth | Data I/O speed, mmap loading, NCCL bandwidth, HBM bandwidth, PCIE bandwidth |
| Backend verification | FlashAttention backend enabled, MoE backend correct, CUDA kernel selection |
| Infrastructure | Checkpoint load success, W&B log connected, sample consumption speed, memory utilization |
| GPU efficiency | MFU, TCA vs target |

#### L1: Numerical Health

| Category | Metrics |
|----------|---------|
| Gradient | No NaN/Inf, gradient distribution normal, no vanishing/exploding |
| Activation | Softmax/Attention distribution reasonable, activation range normal |
| Parameters | Parameter drift from initialization percentage, initialization sanity check |
| Training dynamics | Loss spike detection, residual stream write ratio, MoE entropy |

#### L2: Overfitting Test

- 100-1000 samples, fixed seed, 5-10 epochs
- RecSys: NDCG@10 / Recall@10 -> near 1.0
- LLM: perplexity < 1.1 or loss < 0.01
- Assertion: training loss must monotonically decrease to near 0
- ~10 minutes to complete

#### L3: Domain Metrics

**RecSys specific:**
- Embedding norm stability (user/item embedding L2 norm doesn't explode)
- Popularity bias internal metrics (hot item embedding similarity not too high)
- Negative sampling quality (sampled negatives average score < positive)

**LLM specific:**
- Per-token loss distribution (not concentrated on few tokens)
- Attention entropy (attention not overly concentrated)
- KV cache memory growth linear
- Generation diversity (top-k / nucleus sampling entropy > threshold)

#### L4: End-to-End Pipeline

- Full flow (data -> training -> inference -> evaluation) passes on tiny data

### 3.5 ml-diagnostics

**Adapted from:** systematic-debugging

**Retained:** Phased process (gather evidence -> pattern analysis -> hypothesis testing -> fix), question architecture after 3 failures.

**Three core questions:**

1. **Why not converging?**
   - Loss not decreasing / oscillating / diverging
   - Gradient vanishing / exploding
   - Activation collapse / attention degradation

2. **What early anomaly signals?**
   - Loss spike
   - Abnormal parameter drift from initialization
   - MoE entropy anomaly (load imbalance)
   - Abnormal residual stream write ratio

3. **Why not fast enough?**
   - I/O bottleneck (data reading dragging GPU)
   - Low GPU utilization (MFU/TCA not meeting target)
   - High communication overhead (insufficient NCCL bandwidth)
   - Per-layer decomposition to locate bottleneck substructure

**Evidence collection:** Call toolkit/ tools + skill guides agent to write project-specific diagnostic code.

### 3.6 ml-subagent-dev

**Adapted from:** subagent-driven-development

**Retained:** One subagent per subtask, two-stage review, controller provides full context.

**Changes:**
- Implementer subagent: write unit tests (deterministic parts) -> implement -> run Validation Pyramid
- Spec reviewer: check compliance with experiment design (hypothesis, variable control, no extra changes)
- Quality reviewer: check Validation Pyramid results per layer + code quality
- Each subtask completion records: metric data, conclusion, anomaly log

### 3.7 ml-verification

**Adapted from:** verification-before-completion

**Completion criteria:**
1. All enabled Validation Pyramid layers passed
2. Each subtask has clear conclusion (effective / ineffective / needs further experiment)
3. Summary report: all subtask conclusions + overall judgment + follow-up recommendations
4. Human decides: finish / add subtasks / new brainstorm round

---

## 4. Toolkit Design

**Principle:** Only codify tools that agents struggle to write correctly from scratch and are highly reusable. Toolkit itself tested with traditional TDD.

### 4.1 Tool Responsibilities

**profiling/mfu_calculator.py**
- Input: model, input shape, training step time
- Computes: theoretical FLOPS (auto-calculated from model structure), actual FLOPS, MFU, TCA
- Difficulty: different architectures (dense / MoE / sparse attention) have different theoretical FLOPS formulas; requires hardware spec lookup (GPU peak compute)
- Output: structured result (MFU value, whether meeting target, bottleneck hints)

**profiling/layer_profiler.py**
- Input: model, mock input
- Function: per-layer/segment forward+backward timing breakdown
- Difficulty: correct use of `torch.profiler`, CUDA synchronization timing, warmup handling
- Output: per-layer/segment time, percentage, sorted

**profiling/memory_profiler.py**
- Input: model, mock input
- Function: memory analysis (peak / allocated / reserved / fragmentation)
- Difficulty: distinguishing PyTorch allocated vs CUDA reserved, before/after checkpoint comparison
- Output: memory breakdown, whether obvious waste exists

**monitors/gradient_monitor.py**
- Function: register backward hooks, collect per-layer gradient mean / std / max / min / distribution
- Difficulty: correct hook registration and removal (avoid memory leaks), aggregation under distributed training
- Output: per-layer gradient stats, anomaly detection (automatic vanishing/exploding judgment)

**monitors/activation_monitor.py**
- Function: register forward hooks, collect activation distribution
- Difficulty: same hook management as gradient_monitor + special handling for attention scores, softmax outputs
- Output: per-layer activation stats, distribution anomaly detection

**monitors/loss_tracker.py**
- Function: record per-step loss, detect spikes, trend analysis
- Difficulty: spike threshold judgment (absolute vs relative change vs sliding window), distinguishing normal fluctuation from anomaly
- Output: loss curve summary, spike list, trend judgment (decreasing / stagnating / diverging)

**monitors/parameter_drift.py**
- Function: record initial parameter snapshot, periodically compute drift percentage
- Difficulty: memory efficiency for large models (can't copy all parameters), which layers to monitor
- Output: per-layer parameter drift, anomaly detection

### 4.2 Design Principles

1. **Pure PyTorch dependency** — `import torch` is all you need, no training framework dependency
2. **Plug and play** — each file independent, no cross-dependencies
3. **Structured output** — returns dict/dataclass for easy agent parsing and judgment
4. **Context manager / decorator pattern** — easy to integrate into user's existing code

```python
# Usage example
from toolkit.monitors import gradient_monitor

with gradient_monitor(model) as gm:
    loss = model(x).sum()
    loss.backward()

report = gm.report()
# {'layers': {'attn.weight': {'mean': 0.01, 'std': 0.003, 'status': 'healthy'}, ...}}
```

5. **Non-invasive** — hook registration/removal auto-managed, clean up after use
6. **Toolkit itself uses traditional TDD** — deterministic code, standard pytest tests

---

## 5. Skill and Toolkit Collaboration

### 5.1 Collaboration Model

```
Skill responsible for:              Toolkit responsible for:
  When to use                         How to compute
  How to interpret results            How to collect data
  What to do on failure               Structured output
  What to do next                     Correct API usage
```

**Example flow (MFU check):**

```
1. validation-pyramid skill reads validation config table
   -> L0 engineering efficiency enabled
   -> load vp-engineering-efficiency skill

2. vp-engineering-efficiency skill guides agent:
   "Use toolkit/profiling/mfu_calculator.py to compute MFU"
   "Pass in model and mock input"
   "MFU < user target -> trigger ml-diagnostics"

3. Agent writes integration code:
   from toolkit.profiling.mfu_calculator import calculate_mfu
   result = calculate_mfu(model, input_shape, step_time)

4. Agent interprets result (guided by skill):
   - MFU meets target -> L0 passed, proceed to L1
   - MFU below target -> trigger ml-diagnostics
     -> skill guides: use layer_profiler for per-layer decomposition
     -> locate bottleneck substructure
```

### 5.2 Toolkit Reference Convention in Skills

```markdown
## Check Steps

1. Use `toolkit/profiling/mfu_calculator.py` to compute MFU
   - Input: model instance, input shape, step time
   - Expected: MFU >= user-defined threshold (defined in validation config)

2. If below target, use `toolkit/profiling/layer_profiler.py` for per-layer decomposition
```

**What we don't do:**
- Skills don't contain toolkit implementation details
- Skills don't repeat toolkit docstrings
- Toolkit doesn't contain diagnostic logic (that's the skill's job)

### 5.3 Handling User's Existing Infra

| Scenario | Handling |
|----------|---------|
| User has MFU monitoring | Skill guides agent to use user's tool, skip toolkit |
| User has partial monitoring | Skill guides agent to supplement with toolkit |
| User has no relevant tools | Skill guides agent to integrate toolkit |
| Toolkit discovers user infra issue | Give advice, don't auto-modify |

### 5.4 Framework Adaptation

Toolkit is pure PyTorch. Users may use DeepSpeed / Megatron etc. Adaptation logic lives in frameworks/ skills:

```
frameworks/deepspeed/SKILL.md guides:
  "When using toolkit's gradient_monitor,
   under DeepSpeed ZeRO-3 parameters are sharded,
   need to gather first before monitoring, example..."
```

**Toolkit doesn't handle framework differences. frameworks/ skills tell the agent how to bridge.**

---

## 6. Phased Delivery Plan

**Principle:** Deliver the most critical differentiator first, expand progressively. Each phase is independently usable.

### Phase 1: Foundation (can run)

**Goal:** Fork complete, namespace switched, core workflow skeleton usable.

- Fork superpowers, rename to superpowers-ml
- Rename commands: `ml-brainstorm`, `ml-plan`, `ml-execute`
- Adapt plugin.json / hooks / multi-platform configs for new naming
- `using-superpowers-ml` skill (entry skill, replaces using-superpowers)
- `ml-brainstorming` skill v1 (task type identification, context collection, validation scope confirmation)
- `ml-experiment-planning` skill v1 (subtask decomposition + shared infra annotation)

**Delivery criteria:** Can complete a full brainstorm -> plan flow.

### Phase 2: Validation Pyramid Core

**Goal:** The most critical differentiating capability goes live.

- `validation-pyramid` skill (orchestration layer + dynamic routing + decision tree)
- `vp-engineering-efficiency` skill + toolkit (mfu_calculator, layer_profiler, memory_profiler)
- `vp-numerical-health` skill + toolkit (gradient_monitor, activation_monitor, loss_tracker, parameter_drift)
- `vp-overfitting-test` skill
- Traditional TDD tests for toolkit

**Delivery criteria:** A PyTorch training task can pass L0 + L1 + L2 validation.

### Phase 3: Complete Workflow

**Goal:** End-to-end workflow closed loop.

- `vp-domain-metrics` skill (recsys + llm)
- `vp-e2e-pipeline` skill
- `ml-diagnostics` skill (three core questions + hierarchical decomposition)
- `ml-subagent-dev` skill (ML-adapted review criteria)
- `ml-verification` skill (conclusion summary + recommendations)

**Delivery criteria:** A complete ML experiment can go through the full pipeline: brainstorm -> plan -> execute -> validate -> conclude.

### Phase 4: Framework Knowledge + Extensions

**Goal:** Cover mainstream frameworks, support more task types.

- `frameworks/pytorch` skill
- `frameworks/huggingface` skill
- `frameworks/megatron` skill
- `frameworks/deepspeed` skill
- `frameworks/sglang` skill
- `frameworks/vllm` skill
- `frameworks/wandb` skill
- Validation check sets for dataset preparation, baseline reproduction, inference deployment, etc.
- Community documentation, installation guide

**Delivery criteria:** Different frameworks and task types all have corresponding skill support.

### Phase 5: Continuous Iteration

**Goal:** Continuously improve based on real usage feedback.

- Test and harden each skill using writing-skills TDD methodology
- Discover new kernel metrics from real experiments -> add to pyramid
- Discover skill loopholes from actual agent behavior -> plug them
- Codify new toolkit tools (only when meeting "agent can't write correctly" criteria)
