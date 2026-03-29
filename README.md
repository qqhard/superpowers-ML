# SPML — ML SuperPowers

SPML is an addon plugin for [Superpowers](https://github.com/obra/superpowers) that extends it with ML experiment workflows: Validation Pyramid, experiment-driven development, and Watchdog-based training monitoring.

Superpowers provides the foundation — TDD, code review, subagent architecture, verification. SPML adds the ML domain knowledge on top: what to validate, how to monitor training, and how to draw evidence-based conclusions.

## What makes ML different

In traditional software, code runs = result correct. In ML, code runs without errors does NOT mean the result is correct.

**"Not working" is reasonable in ML, but the process must be correct.** If an implementation error causes poor results, you may misjudge your experimental strategy as ineffective, wasting an entire research direction.

SPML addresses this with:
- **Validation Pyramid** — 2-level verification (static analysis, runtime + pipeline validation) that separates "implementation bug" from "strategy doesn't work"
- **Watchdog Agent** — active monitoring of long-running training with auto-restart, parameter fixing, and sub-agent spawning for complex issues
- **Experiment-driven workflow** — hypothesis, independent/dependent/control variables, conclusion recording with metric evidence

## Installation

### Prerequisites

Install [Superpowers](https://github.com/obra/superpowers) first. SPML depends on Superpowers for general development skills (TDD, code review, debugging, etc.).

### Claude Code

In Claude Code:

```
/plugin marketplace add qqhard/superpowers-ML
/plugin install spml
```

### Codex

SPML also works with Codex through native skill discovery.

Install `superpowers` first, then install SPML:

```bash
git clone https://github.com/obra/superpowers.git ~/.codex/superpowers
git clone https://github.com/qqhard/superpowers-ML.git ~/.codex/spml
mkdir -p ~/.agents/skills
ln -s ~/.codex/superpowers/skills ~/.agents/skills/superpowers
ln -s ~/.codex/spml/skills ~/.agents/skills/spml
```

See [docs/README.codex.md](docs/README.codex.md) for the full Codex guide.

### Verify Installation

Start a new session and check that both skill sets are available.

Claude Code:

```
/superpowers:brainstorm   → general software brainstorming
/spml:brainstorm          → ML experiment brainstorming
```

Codex:

Ask Codex to use `superpowers:brainstorming` for general software work or
`spml:ml-brainstorming` for ML experiment work.

## How the two plugins work together

```
General software development:
  /superpowers:brainstorm → superpowers:writing-plans → superpowers:subagent-driven-development
  All skills from Superpowers, SPML not involved.

ML experiments:
  /spml:brainstorm → spml:experiment-planning → spml:ml-subagent-dev
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
ml-subagent-dev
    Execute each subtask: unit test → implement → Validation Pyramid
    |
training-handoff
    Generate training script + Watchdog prompt + experiment context
    |
watchdog (independent session)
    Active monitoring: auto-restart, parameter fixing, anomaly diagnosis
    |
training-resume (independent session)
    Analyze results or diagnose issues, decide next step
    |
verification
    Evidence-based conclusion: effective / ineffective / inconclusive
```

### Validation Pyramid

Each subtask passes through 2 levels of validation before claiming correctness:

| Level | What it checks | Time |
|-------|---------------|------|
| **L0: Static Analysis** | Device consistency, precision config, FlashAttention, optimizer, DataLoader, logging & observability + 15 advisory checks | Seconds |
| **L1: Runtime Validation** | Train ~5 min collecting MFU, TCA, throughput, gradient health, loss trend, then verify full pipeline: checkpoint → inference → evaluation | ~5-15 min |

L0 runs as a subagent (code review style). L1 runs as a skill invoked by the orchestrator. L0 must pass before L1.

### Watchdog Agent

Long-running training is monitored by an independent agent session with three operating modes:

- **Monitor** — report only, no intervention
- **Guardian** (default) — auto-restart on environment failures, auto-fix simple parameter problems, report complex issues
- **Autonomous** — handle everything including complex issues via sub-agent spawning

Problems are classified into 3 tiers: environment problems (restart), simple parameter problems (fix + restart), and complex problems (sub-agent or report). The watchdog produces a recovery or completion prompt for the next session.

## Skills

### ML Workflow

| Skill | Purpose |
|-------|---------|
| **brainstorming** | Experiment design, context collection, validation scope confirmation |
| **experiment-planning** | Subtask decomposition with validation criteria |
| **data-preparation** | TDD-first dataset processing: validate on small-scale, then full-scale |
| **ml-subagent-dev** | Execute subtasks with VP integration and experiment-aware review |
| **diagnostics** | Systematic diagnosis: why not converging, early anomalies, efficiency bottlenecks |
| **verification** | Evidence-based conclusion with experiment summary |
| **training-handoff** | Generate training script + Watchdog prompt + experiment context |
| **watchdog** | Active monitoring of long-running tasks with 3 operating modes |
| **training-resume** | Recovery or completion entry point after long-running tasks |

### Validation Pyramid

| Skill | Checks |
|-------|--------|
| **validation-pyramid** | 2-level validation orchestration integrated into ml-subagent-dev workflow |
| **ml-static-checks** | L0: Static analysis — device consistency, precision, FA, optimizer, DataLoader, logging & observability + 15 advisory checks |
| **ml-runtime-validator** | L1: Runtime validation — train ~5 min with metrics, then verify full pipeline (checkpoint, inference, evaluation) |

### Shared Infrastructure (modified from Superpowers)

| Skill | Why modified |
|-------|-------------|
| **executing-plans** | Routes to `spml:experiment-planning` instead of `superpowers:writing-plans` |

### From Superpowers (not included, used via cross-plugin reference)

TDD, systematic-debugging, brainstorming, writing-plans, dispatching-parallel-agents, using-git-worktrees, requesting/receiving-code-review, finishing-a-development-branch, verification-before-completion, writing-skills — all provided by [Superpowers](https://github.com/obra/superpowers).

## Toolkit

Profiling tools that agents struggle to write correctly from scratch:

| Tool | Purpose |
|------|---------|
| `toolkit/profiling/l0_runner.py` | L1 runtime validation entry point — orchestrates metric collection |
| `toolkit/profiling/mfu_calculator.py` | Theoretical FLOPS + MFU/TCA calculation |
| `toolkit/profiling/dcgm_profiler.py` | NVIDIA DCGM field 1004 profiling for TCA measurement |
| `toolkit/profiling/gap_analyzer.py` | Hierarchical bottleneck decomposition |
| `toolkit/profiling/layer_profiler.py` | Per-layer forward/backward timing |
| `toolkit/profiling/memory_profiler.py` | Memory analysis and fragmentation |

## Acknowledgments

SPML builds on [Superpowers](https://github.com/obra/superpowers) by [Jesse Vincent](https://github.com/obra). The skill system architecture, workflow patterns, and multi-platform support are all from Superpowers. Read more: [Superpowers for Claude Code](https://blog.fsck.com/2025/10/09/superpowers/).

## License

MIT License — see [LICENSE](LICENSE) file for details.
