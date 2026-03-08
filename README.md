# SPML — ML SuperPowers

SPML is a complete ML/RecSys/LLM training development workflow for AI coding agents. Built on composable "skills" that guide agents through experiment design, validation, long-running training monitoring, and conclusion analysis.

Forked from [Superpowers](https://github.com/obra/superpowers) by [Jesse Vincent](https://github.com/obra). Superpowers provides a battle-tested skill system for AI coding agents — SPML extends it with ML-specific workflows: the Validation Pyramid, experiment-driven development, and Watchdog-based training monitoring. Thank you Jesse for building the foundation that made this possible.

## What makes ML different

In traditional software, code runs = result correct. In ML, code runs without errors does NOT mean the result is correct.

**"Not working" is reasonable in ML, but the process must be correct.** If an implementation error causes poor results, you may misjudge your experimental strategy as ineffective, wasting an entire research direction.

SPML addresses this with:
- **Validation Pyramid** — layered verification (engineering efficiency, process metrics, overfitting test, e2e pipeline) that separates "implementation bug" from "strategy doesn't work"
- **Watchdog Agent** — read-only monitoring of long-running training through independent agent sessions, replacing the human who sits watching W&B dashboards
- **Experiment-driven workflow** — hypothesis, independent/dependent/control variables, conclusion recording with metric evidence

## Installation

> **Note:** SPML is in early development. Installation is manual clone + configuration. Marketplace publishing is planned for later.

### Claude Code

```bash
# Clone the repo
git clone https://github.com/qqhard/superpowers-ML.git ~/.claude/plugins/spml

# Register as local plugin
claude plugin add -- ~/.claude/plugins/spml
```

### Cursor

```bash
# Clone the repo
git clone https://github.com/qqhard/superpowers-ML.git ~/.cursor/plugins/spml
```

Then in Cursor settings, add the plugin path.

### Codex

```bash
git clone https://github.com/qqhard/superpowers-ML.git ~/.codex/spml
mkdir -p ~/.agents/skills
ln -s ~/.codex/spml/skills ~/.agents/skills/spml
```

Restart Codex to discover the skills.

### OpenCode

```bash
git clone https://github.com/qqhard/superpowers-ML.git ~/.config/opencode/spml
mkdir -p ~/.config/opencode/skills
ln -s ~/.config/opencode/spml/skills ~/.config/opencode/skills/spml
```

Restart OpenCode to discover the skills.

### Verify Installation

Start a new session and describe an ML task (e.g., "help me design a training experiment" or "let's optimize this model's MFU"). The agent should invoke the relevant SPML skill.

### Updating

```bash
cd <install-path> && git pull
```

## The ML Workflow

```
ml-brainstorming
    Refine hypothesis, collect context, confirm validation scope
    |
ml-experiment-planning
    Break into atomic subtasks with validation criteria
    |
ml-subagent-dev
    Execute each subtask: unit test → implement → Validation Pyramid
    |
ml-training-handoff
    Generate training script + Watchdog prompt + experiment context
    |
ml-watchdog (independent session)
    Read-only monitoring, anomaly detection, diagnosis reporting
    |
ml-training-resume (independent session)
    Analyze results or diagnose issues, decide next step
    |
ml-verification
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

Users can skip any layer. The pyramid dynamically loads checks based on architecture (Transformer, MoE, CNN) and task type (RecSys, LLM, CV).

### Watchdog Agent

Long-running training is monitored by an independent agent session:

- **Read-only** — observes JSONL metrics log, never intervenes
- **Adaptive frequency** — checks often at start, less in steady state, more near completion
- **Framework-agnostic** — the Watchdog prompt works with any LLM agent (Claude Code, Codex, Cursor, etc.)
- **Session chain** — on anomaly, produces a recovery prompt; on completion, produces a completion prompt. Each prompt starts a fresh session with full context via `experiment-context.md`

Training scripts are core code — independently deployable to production, zero agent dependency.

## Skills Library

### ML-Specific Skills

| Skill | Purpose |
|-------|---------|
| **ml-brainstorming** | Experiment design, context collection, validation scope confirmation |
| **ml-experiment-planning** | Subtask decomposition with validation criteria |
| **ml-data-preparation** | TDD-first dataset processing: validate on small-scale, then full-scale |
| **ml-subagent-dev** | Execute subtasks with VP integration and experiment-aware review |
| **ml-diagnostics** | Systematic diagnosis: why not converging, early anomalies, efficiency bottlenecks |
| **ml-verification** | Evidence-based conclusion with experiment summary |
| **ml-training-handoff** | Generate training script + Watchdog prompt + experiment context |
| **ml-watchdog** | Read-only monitoring of long-running tasks |
| **ml-training-resume** | Recovery or completion entry point after long-running tasks |
| **validation-pyramid** | Layered validation orchestration with dynamic routing |

### VP Layer Skills

| Skill | Checks |
|-------|--------|
| **vp-engineering-efficiency** | MFU, GPU utilization, backend, bandwidth, memory |
| **vp-process-metrics** | Gradients, activations, parameter drift, architecture-specific signals |
| **vp-overfitting-test** | Small-scale overfit with trend-based criteria |
| **vp-e2e-pipeline** | End-to-end data → train → infer → evaluate |

### Inherited from Superpowers

| Skill | Purpose |
|-------|---------|
| **test-driven-development** | RED-GREEN-REFACTOR cycle |
| **systematic-debugging** | 4-phase root cause process |
| **brainstorming** | Socratic design refinement |
| **writing-plans** | Detailed implementation plans |
| **executing-plans** | Batch execution with checkpoints |
| **subagent-driven-development** | Fast iteration with two-stage review |
| **dispatching-parallel-agents** | Concurrent subagent workflows |
| **using-git-worktrees** | Parallel development branches |
| **requesting-code-review** / **receiving-code-review** | Code review workflows |
| **finishing-a-development-branch** | Merge/PR decision workflow |
| **verification-before-completion** | Evidence before claims |
| **writing-skills** | Create and test new skills |

## Toolkit

Profiling tools that agents struggle to write correctly from scratch:

| Tool | Purpose |
|------|---------|
| `toolkit/profiling/mfu_calculator.py` | Theoretical FLOPS + MFU/TCA calculation |
| `toolkit/profiling/layer_profiler.py` | Per-layer forward/backward timing |
| `toolkit/profiling/memory_profiler.py` | Memory analysis and fragmentation |

Pure PyTorch, plug and play, non-invasive. Monitors (gradient, activation, loss tracking) are guided by skills — agents write these per-project.

## Design

For the full design document, see [docs/plans/2026-03-06-superpowers-ml-design.md](docs/plans/2026-03-06-superpowers-ml-design.md).

Key design principles:
1. Code runs ≠ result correct in ML
2. Process must be correct — separate implementation bugs from strategy failures
3. Validation Pyramid extends TDD to ML's non-deterministic domain
4. Test/validation code and core code are strictly separated
5. Training scripts are core code — zero agent dependency, production-deployable
6. Only codify toolkit code agents struggle to write correctly
7. Watchdog protocol is framework-agnostic — any LLM agent can execute it

## Acknowledgments

SPML is a fork of [Superpowers](https://github.com/obra/superpowers) by [Jesse Vincent](https://github.com/obra). The skill system architecture, workflow patterns (brainstorming, TDD, subagent-driven development, git worktrees), and multi-platform support are all inherited from Superpowers. Read more about the original project: [Superpowers for Claude Code](https://blog.fsck.com/2025/10/09/superpowers/).

## License

MIT License — see [LICENSE](LICENSE) file for details. Original copyright (c) 2025 Jesse Vincent.

## Contributing

1. Fork the repository
2. Create a branch for your skill
3. Follow the `writing-skills` skill for creating and testing new skills
4. Submit a PR

## Support

- **Issues**: https://github.com/qqhard/superpowers-ML/issues
- **Upstream**: https://github.com/obra/superpowers
