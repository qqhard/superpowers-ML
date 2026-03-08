# MLSP Design Document

> **For Claude:** This is a design document for a new project. Use superpowers:writing-plans to create the implementation plan.

**Project:** mlsp

**Source:** Fork from [obra/superpowers](https://github.com/obra/superpowers), independent development

**Date:** 2026-03-06

---

## 1. Project Positioning

**Name:** mlsp

**Positioning:** ML/RecSys/LLM training development workflow framework for AI agents. Fork from Superpowers, restructuring all core workflows for ML development.

**Target users:** AI coding agents (Claude Code, Codex, OpenCode, Cursor), serving ML engineering teams, later expanding to community.

**Supported platforms:** Maintain existing Claude Code / Codex / OpenCode / Cursor support.

**Core principles:**

1. In traditional software, code runs = result correct. In ML, code runs ≠ result correct.
2. **"Not working" is reasonable in ML, but the process must be correct.** If an implementation error causes poor results, you may misjudge the experimental strategy itself as ineffective, wasting an entire research direction.
3. Extend TDD with a **Validation Pyramid** — TDD's RED-GREEN-REFACTOR rhythm applies at all levels: traditional unit tests for deterministic code (functions, operators), Pyramid layers for non-deterministic process validation (training efficiency, convergence). Each layer's validation is minute-level, satisfying fast feedback loops. Multi-layer process metrics separate "implementation error" from "strategy ineffective."
5. Users can skip any validation layer — scope confirmed during brainstorm phase.
6. Validation Pyramid dynamically orchestrates based on architecture type, task type, and user context.
7. Only codify toolkit code that agents struggle to write correctly from scratch and is highly reusable; everything else is guided by skills for agents to write on the spot.
8. **Test/validation code and core code are strictly separated.** Core code (model, training, data pipeline) must be zero-test-dependency, production-deployable. All Validation Pyramid checks, monitors, and profiling are external — they observe but never invade core code. After Agentic Engineering completes, core code can be extracted and deployed to production as-is, clean and executable.
9. **Training scripts are core code.** Like model and training loop code, generated training scripts are independently deployable to production with zero agent dependency. The Watchdog monitoring protocol (prompt + experiment-context.md + JSONL logs) is framework-agnostic — any LLM agent can execute it.

---

## 2. Project Structure

```
mlsp/
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
    vp-process-metrics/            # All process/kernel metrics (universal + architecture-specific)
      SKILL.md
      gradient-checks.md           # Universal
      activation-checks.md         # Transformer: attention; general: activations
      parameter-drift.md           # Parameter drift, loss spike
      residual-stream.md           # Residual connection architectures
      moe-checks.md                # MoE: entropy, load balance
      embedding-checks.md          # RecSys: norm stability, popularity bias, negative sampling
      token-loss-checks.md         # LLM: per-token loss, generation diversity
      kv-cache-checks.md           # LLM: KV cache memory growth
    vp-overfitting-test/
      SKILL.md
    vp-e2e-pipeline/
      SKILL.md

    # Restructured workflow skills
    ml-diagnostics/                # Non-convergence / early anomaly / efficiency bottleneck
    ml-subagent-dev/               # ML-adapted subagent execution + review
    ml-verification/               # Completion criteria = pyramid passed + valid conclusions

    # Long-running task monitoring (session chain)
    ml-training-handoff/             # Main Agent → Watchdog: generates training script + context + prompts
    ml-watchdog/                     # Watchdog Agent: read-only monitoring of long-running tasks
    ml-training-resume/              # Recovery/Completion Agent: reads context, decides next workflow step

    # Pluggable framework knowledge (async loaded, created on demand)
    frameworks/                    # Empty initially; add specific framework skills as needed in practice

    # Reused skills (minor adjustments or direct reuse)
    using-mlsp/
    writing-skills/
    dispatching-parallel-agents/
    finishing-a-development-branch/
    receiving-code-review/
    requesting-code-review/

  # === Toolkit (only tools agents struggle to write correctly) ===
  toolkit/
    profiling/
      mfu_calculator.py            # Theoretical FLOPS + measurement + MFU/TCA
      layer_profiler.py            # Per-layer forward/backward timing breakdown
      memory_profiler.py           # Memory analysis
    # monitors/ — not included initially; skills guide agent to write these.
    # Promote to toolkit only when real usage shows agents consistently get them wrong.

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
    Output: experiment design (includes validation scope + long-running confirmation)
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
    |       L0 -> L1 -> L2 -> L3
    |       Any layer fails -> ml-diagnostics
    |       Diagnostics support hierarchical decomposition (whole -> substructure -> operator)
    +-- 4. Spec review (does implementation match experiment design?)
    +-- 5. Record conclusion (metric data + effective/ineffective/needs further study)
    |
    Task needs long-running phase?
    |
    +-- No -> ml-verification (all subtasks complete)
    |
    +-- Yes -> ml-training-handoff
                |
                Output: training script + logs + experiment-context.md + watchdog-prompt.md
                |
                User chooses: separated execution or combined execution
                |
                [User starts training + Watchdog session]
                |
              ml-watchdog (independent session, read-only monitoring)
                |
                +-- Normal completion -> completion-prompt.md
                +-- Anomaly detected -> recovery-prompt.md
                |
              ml-training-resume (independent session)
                |
                +-- From completion -> ml-verification
                +-- From recovery -> back to appropriate stage
                    (code fix / replan / rebrainstorm)
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

**New: Validation scope confirmation (first step)**

No task type classification. Instead, directly ask the user which validation layers apply to their current task. Different tasks naturally lead to different layers being enabled or skipped — an experiment may need L0-L3, dataset prep may only need data quality checks, pipeline work may only need L0+L4. The pyramid's dynamic selection handles this without an extra categorization layer.

**New: Experiment design (when applicable)**
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

**Output:** Design document with validation scope described in natural language. No formal YAML schema required — the agent records decisions from the brainstorm conversation naturally in the design doc (e.g., "L0: check MFU and NCCL bandwidth, skip I/O since user has existing pipeline. L1: gradient + attention + MoE entropy. L2: overfit test. L3: recsys metrics. L4: e2e on tiny data.").

### 3.2 ml-experiment-planning

**Adapted from:** writing-plans

**Retained:** Bite-sized steps, exact file paths, complete code, executable commands.

**Code separation principle:**

Core code (model, training, data) must never import from test/validation code or toolkit. Validation scripts observe core code externally via hooks/wrappers. After development, core code can be extracted and deployed to production as-is. The specific directory layout adapts to the user's existing project structure — the agent determines where to place test and validation code during brainstorm based on what the user already has.

**Output structure:**

```markdown
# [Experiment Name] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use mlsp:ml-subagent-dev

**Goal:** [one sentence]
**Hypothesis:** [doing X is expected to cause Y]
**Validation scope:** [reference validation scope from brainstorm design doc]

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

**Extends:** test-driven-development

**Orchestration logic:** Based on validation scope from brainstorm design doc, execute enabled checks in L0->L4 order, pass each layer before entering next, failure triggers ml-diagnostics.

**Three granularity levels:**

| Granularity | Applicable Scenario | Method |
|-------------|-------------------|--------|
| Function/operator | Custom loss, custom layer, single operator efficiency | Traditional unit test, deterministic assertions |
| Module/layer | Network substructure efficiency, segmented validation | Mock data, segments defined in brainstorm |
| Experiment | L0-L3 full pyramid | Training process metrics |

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

#### L1: Process Metrics (universal + architecture-specific, loaded on demand)

**Universal (always applicable):**

| Category | Metrics |
|----------|---------|
| Gradient | No NaN/Inf, gradient distribution normal, no vanishing/exploding |
| Parameters | Parameter drift from initialization percentage, initialization sanity check |
| Training dynamics | Loss spike detection |

**Architecture-specific (loaded based on brainstorm context):**

| Architecture/Domain | Metrics |
|-------------------|---------|
| Transformer | Attention distribution, attention entropy |
| Residual networks | Residual stream write ratio |
| MoE | MoE entropy, load balance |
| RecSys | Embedding norm stability, popularity bias, negative sampling quality |
| LLM | Per-token loss distribution, generation diversity, KV cache memory growth |

Additional architecture-specific checks can be added as new sub-files over time.

#### L2: Overfitting Test

- 100-1000 samples, fixed seed, 5-10 epochs
- Assertion: training loss decreases steadily and quickly (consistent downward trend)
- Task-specific metrics show clear improvement trend
- No absolute threshold required — the test validates the trend, not the final value
- ~10 minutes to complete

#### L3: End-to-End Pipeline

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

### 3.8 ml-training-handoff

**New skill.** Bridges VP validation (minute-level) and long-running execution (hours/days).

**Trigger:** After ml-subagent-dev completes subtasks and VP passes, when task includes long-running phase.

**Produces:**
1. Training script — core code, zero agent dependency, production-deployable. Includes dual output logging: tqdm progress bar (terminal, for humans) + JSONL metrics file (for Watchdog).
2. experiment-context.md — full experiment context: design, VP baseline, training config, code state.
3. watchdog-prompt.md — short, framework-agnostic prompt for starting a Watchdog session.

**User options:**
- Separated execution: user runs training independently, starts Watchdog in a separate session
- Combined execution: user starts Watchdog session which launches training and monitors

### 3.9 ml-watchdog

**New skill.** Watchdog Agent behavior definition — read-only monitoring of long-running tasks.

**Mechanism:** Periodically reads JSONL metrics log, compares against VP baseline from experiment-context.md, detects anomalies through trend analysis and pattern recognition (not just threshold alerts).

**Adaptive frequency:** High at start (catch startup issues), normal during steady state, higher near completion. User can override anytime.

**Output depends on execution mode:**
- Separated: silent when normal, speaks only on anomaly
- Combined: periodic progress reports + anomaly diagnostics

**On anomaly:** Writes diagnosis to experiment-context.md, produces recovery-prompt.md.
**On completion:** Writes summary to experiment-context.md, produces completion-prompt.md.

**Boundary:** Observes only. Never stops training, modifies code, or adjusts hyperparameters.

### 3.10 ml-training-resume

**New skill.** Recovery or Completion Agent entry point.

**From recovery:** Agent reads experiment-context.md (including Watchdog diagnosis), autonomously judges which workflow stage to return to: code fix, hyperparameter adjustment, replan, or rebrainstorm.

**From completion:** Agent analyzes final results vs hypothesis, enters ml-verification flow.

**Key:** Agent decides rollback level from evidence, not from Watchdog suggestion.

---

## 4. Toolkit Design

**Principle:** Only codify tools that agents struggle to write correctly from scratch and are highly reusable. Toolkit itself tested with traditional TDD.

### 4.1 Tool Responsibilities (Phase 2 scope: profiling only)

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

**monitors/ — deferred.** Gradient monitoring, activation tracking, loss spike detection, and parameter drift are guided by skills. Agent writes these per-project. If real usage shows agents consistently get them wrong, promote to toolkit.

### 4.2 Design Principles

1. **Pure PyTorch dependency** — `import torch` is all you need, no training framework dependency
2. **Plug and play** — each file independent, no cross-dependencies
3. **Structured output** — returns dict/dataclass for easy agent parsing and judgment
4. **Context manager / decorator pattern** — easy to integrate into user's existing code

```python
# Usage example
from toolkit.profiling.mfu_calculator import calculate_mfu

result = calculate_mfu(model, input_shape=(8, 512), step_time_ms=150.0)
# {'mfu': 0.42, 'tca': 0.51, 'theoretical_tflops': 312.5, 'status': 'below_target'}
```

5. **Non-invasive** — hook registration/removal auto-managed, clean up after use. Toolkit attaches externally to models (via PyTorch hooks, wrappers, profiler context managers) — core code never imports or depends on toolkit. After validation, remove all toolkit references and core code is production-ready as-is.
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
1. validation-pyramid skill reads validation scope from design doc
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

- Fork superpowers, rename to mlsp
- Rename commands: `ml-brainstorm`, `ml-plan`, `ml-execute`
- Adapt plugin.json / hooks / multi-platform configs for new naming
- `using-mlsp` skill (entry skill, replaces using-superpowers)
- `ml-brainstorming` skill v1 (context collection, validation scope confirmation)
- `ml-experiment-planning` skill v1 (subtask decomposition + shared infra annotation)

**Delivery criteria:** Can complete a full brainstorm -> plan flow.

### Phase 2: Validation Pyramid Core

**Goal:** The most critical differentiating capability goes live.

- `validation-pyramid` skill (orchestration layer + dynamic routing + decision tree)
- `vp-engineering-efficiency` skill + toolkit (mfu_calculator, layer_profiler, memory_profiler)
- `vp-process-metrics` skill (universal + architecture-specific sub-files; guides agent to write monitors per-project)
- `vp-overfitting-test` skill
- Traditional TDD tests for toolkit

**Delivery criteria:** A PyTorch training task can pass L0 (engineering) + L1 (process metrics) + L2 (overfit test) validation.

### Phase 3: Complete Workflow

**Goal:** End-to-end workflow closed loop.

- `vp-e2e-pipeline` skill
- `ml-diagnostics` skill (three core questions + hierarchical decomposition)
- `ml-subagent-dev` skill (ML-adapted review criteria)
- `ml-verification` skill (conclusion summary + recommendations)
- `ml-training-handoff` skill (generates training script + context + Watchdog prompt)
- `ml-watchdog` skill (read-only long-running task monitoring)
- `ml-training-resume` skill (recovery/completion entry point, rollback decision)

**Delivery criteria:** A complete ML experiment can go through the full pipeline: brainstorm -> plan -> execute -> validate -> long-running training with monitoring -> conclude.

### Phase 4: Framework Knowledge + Extensions

**Goal:** Add framework skills and expand coverage based on real usage.

- Create framework skills on demand (e.g., `frameworks/deepspeed` when a user first needs DeepSpeed guidance)
- Promote monitors to toolkit if agents consistently write them incorrectly
- Community documentation, installation guide

**Delivery criteria:** Most common workflows in the team's daily practice have skill coverage.

### Phase 5: Continuous Iteration

**Goal:** Continuously improve based on real usage feedback.

- Test and harden each skill using writing-skills TDD methodology
- Discover new kernel metrics from real experiments -> add to pyramid
- Discover skill loopholes from actual agent behavior -> plug them
- Codify new toolkit tools (only when meeting "agent can't write correctly" criteria)

---

## 7. Simplification Decisions

Things we deliberately chose NOT to build upfront, to avoid over-engineering:

| Decision | Rationale |
|----------|-----------|
| No formal YAML schema for validation config | Agent records brainstorm decisions in natural language in the design doc. A rigid schema adds complexity and format errors without real benefit. |
| Toolkit starts with profiling only (3 files), no monitors | loss_tracker, parameter_drift, gradient/activation monitors are simple enough for agents to write. Promote to toolkit only if real usage shows agents consistently get them wrong. |
| No prescribed user project directory structure | Users have existing project layouts. We enforce the principle (core code never imports test/validation code) not a specific directory template. Agent adapts to user's structure. |
| No pre-planned framework skill list | frameworks/ starts empty. Create specific framework skills on demand when actually needed, not 7 placeholder directories. |
| No task type classification table | The pyramid's "skip layers that don't apply" mechanism already handles different task types. An explicit 6-category taxonomy is a redundant abstraction layer. |
