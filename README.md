# SPML — ML SuperPowers

SPML is an addon plugin for [Superpowers](https://github.com/obra/superpowers) that extends it with ML experiment workflows: Validation Pyramid, experiment-driven development, and Watchdog-based training monitoring.

Superpowers provides the foundation — TDD, code review, subagent architecture, verification. SPML adds the ML domain knowledge on top: what to validate, how to monitor training, and how to draw evidence-based conclusions.

## What makes ML different

In traditional software, code runs = result correct. In ML, code runs without errors does NOT mean the result is correct.

**"Not working" is reasonable in ML, but the process must be correct.** If an implementation error causes poor results, you may misjudge your experimental strategy as ineffective, wasting an entire research direction.

SPML addresses this with:
- **Validation Pyramid** — layered verification (engineering efficiency, process metrics, overfitting test, e2e pipeline) that separates "implementation bug" from "strategy doesn't work"
- **Watchdog Agent** — read-only monitoring of long-running training through independent agent sessions
- **Experiment-driven workflow** — hypothesis, independent/dependent/control variables, conclusion recording with metric evidence

## Installation

### Prerequisites

Install [Superpowers](https://github.com/obra/superpowers) first. SPML depends on Superpowers for general development skills (TDD, code review, debugging, etc.).

### Claude Code

In Claude Code:

```
/plugin marketplace add qqhard/superpowers-ML
/plugin install spml@spml-dev
```

### Verify Installation

Start a new session and check that both plugin namespaces are available:

```
/superpowers:brainstorm   → general software brainstorming
/spml:brainstorm          → ML experiment brainstorming
```

### Updating

```
/plugin update spml
```

## How the two plugins work together

```
General software development:
  /superpowers:brainstorm → superpowers:writing-plans → superpowers:subagent-driven-development
  All skills from Superpowers, SPML not involved.

ML experiments:
  /spml:brainstorm → spml:experiment-planning → spml:subagent-dev
  ML workflow from SPML, general discipline (TDD, code review) from Superpowers.
```

SPML skills reference Superpowers skills where needed (e.g., `superpowers:finishing-a-development-branch`, `superpowers:using-git-worktrees`). Cross-plugin skill invocation works transparently.

## The ML Workflow

```
brainstorming
    Refine hypothesis, collect context, confirm validation scope
    |
experiment-planning
    Break into atomic subtasks with validation criteria
    |
subagent-dev
    Execute each subtask: unit test → implement → Validation Pyramid
    |
training-handoff
    Generate training script + Watchdog prompt + experiment context
    |
watchdog (independent session)
    Read-only monitoring, anomaly detection, diagnosis reporting
    |
training-resume (independent session)
    Analyze results or diagnose issues, decide next step
    |
verification
    Evidence-based conclusion: effective / ineffective / inconclusive
```

### Validation Pyramid

Each subtask passes through layered validation before claiming correctness:

| Layer | What it checks | Time |
|-------|---------------|------|
| **L0: Engineering Efficiency** | MFU, GPU utilization, backend verification, I/O bandwidth | Minutes |
| **L1: Process Metrics** | Gradient health, activation patterns, architecture-specific signals | Minutes |
| **L2: Overfitting Test** | Loss decreases on 100-1000 samples, fixed seed | ~10 min |
| **L3: E2E Pipeline** | Full flow on tiny data: data → train → infer → evaluate | Minutes |

The pyramid dynamically loads checks based on architecture (Transformer, MoE, CNN) and task type.

### Watchdog Agent

Long-running training is monitored by an independent agent session:

- **Read-only** — observes JSONL metrics log, never intervenes
- **Adaptive frequency** — checks often at start, less in steady state, more near completion
- **Session chain** — on anomaly, produces a recovery prompt; on completion, produces a completion prompt

## Skills

### ML Workflow

| Skill | Purpose |
|-------|---------|
| **brainstorming** | Experiment design, context collection, validation scope confirmation |
| **experiment-planning** | Subtask decomposition with validation criteria |
| **data-preparation** | TDD-first dataset processing: validate on small-scale, then full-scale |
| **subagent-dev** | Execute subtasks with VP integration and experiment-aware review |
| **diagnostics** | Systematic diagnosis: why not converging, early anomalies, efficiency bottlenecks |
| **verification** | Evidence-based conclusion with experiment summary |
| **training-handoff** | Generate training script + Watchdog prompt + experiment context |
| **watchdog** | Read-only monitoring of long-running tasks |
| **training-resume** | Recovery or completion entry point after long-running tasks |

### Validation Pyramid

| Skill | Checks |
|-------|--------|
| **validation-pyramid** | Layered validation orchestration with dynamic routing |
| **vp-engineering-efficiency** | MFU, GPU utilization, backend, bandwidth, memory |
| **vp-process-metrics** | Gradients, activations, parameter drift, architecture-specific signals |
| **vp-overfitting-test** | Small-scale overfit with trend-based criteria |
| **vp-e2e-pipeline** | End-to-end data → train → infer → evaluate |

### Shared Infrastructure (modified from Superpowers)

| Skill | Why modified |
|-------|-------------|
| **executing-plans** | Routes to `spml:experiment-planning` instead of `superpowers:writing-plans` |
| **subagent-driven-development** | Routes to `spml:experiment-planning` instead of `superpowers:writing-plans` |

### From Superpowers (not included, used via cross-plugin reference)

TDD, systematic-debugging, brainstorming, writing-plans, dispatching-parallel-agents, using-git-worktrees, requesting/receiving-code-review, finishing-a-development-branch, verification-before-completion, writing-skills — all provided by [Superpowers](https://github.com/obra/superpowers).

## Toolkit

Profiling tools that agents struggle to write correctly from scratch:

| Tool | Purpose |
|------|---------|
| `toolkit/profiling/mfu_calculator.py` | Theoretical FLOPS + MFU/TCA calculation |
| `toolkit/profiling/layer_profiler.py` | Per-layer forward/backward timing |
| `toolkit/profiling/memory_profiler.py` | Memory analysis and fragmentation |

## Acknowledgments

SPML builds on [Superpowers](https://github.com/obra/superpowers) by [Jesse Vincent](https://github.com/obra). The skill system architecture, workflow patterns, and multi-platform support are all from Superpowers. Read more: [Superpowers for Claude Code](https://blog.fsck.com/2025/10/09/superpowers/).

## License

MIT License — see [LICENSE](LICENSE) file for details.
