# SPML — ML SuperPowers

SPML is an addon plugin for [Superpowers](https://github.com/obra/superpowers) that extends it with ML experiment workflows: Validation Pyramid, experiment-driven development, Watchdog-based training monitoring, and Auto Research — an autonomous iteration loop where a Supervisor dispatches Researcher subagents, runs evaluation, manages git, and accumulates experience across rounds.

Superpowers provides the foundation — TDD, code review, subagent architecture, verification. SPML adds the ML domain knowledge on top: what to validate, how to monitor training, how to draw evidence-based conclusions, and how to run a research loop autonomously while a human stays on the loop.

## What makes ML different

In traditional software, code runs = result correct. In ML, code runs without errors does NOT mean the result is correct.

**"Not working" is reasonable in ML, but the process must be correct.** If an implementation error causes poor results, you may misjudge your experimental strategy as ineffective, wasting an entire research direction.

SPML addresses this with:
- **Validation Pyramid** — 2-level verification (static analysis, runtime + pipeline validation) that separates "implementation bug" from "strategy doesn't work"
- **Watchdog Agent** — active monitoring of long-running training with auto-restart, parameter fixing, and sub-agent spawning for complex issues
- **Auto Research** — protocol-driven autonomous iteration: Supervisor dispatches fresh Researcher subagents each round, runs the fixed eval script, commits improvements and rolls back regressions, and passes lessons between rounds through an experiences log
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

Ask Claude to use `superpowers:brainstorming` for general software work,
`spml:ml-brainstorming` for an ML experiment, or `spml:autoresearch-create`
to start an Auto Research project.

Codex:

Ask Codex to use `superpowers:brainstorming` for general software work,
`spml:ml-brainstorming` for ML experiment work, or `spml:autoresearch-create`
to start an Auto Research project.

## How the two plugins work together

```
General software development:
  /superpowers:brainstorm → superpowers:writing-plans → superpowers:subagent-driven-development
  All skills from Superpowers, SPML not involved.

ML experiments:
  spml:ml-brainstorming → spml:experiment-planning → spml:ml-subagent-dev
  ML workflow from SPML, general discipline (TDD, code review) from Superpowers.

Auto Research (autonomous iteration after a validated baseline):
  spml:autoresearch-create → spml:ml-brainstorming (autoresearch mode) →
  spml:experiment-planning → spml:ml-subagent-dev →
  spml:autoresearch-handoff → spml:autoresearch-run
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
    ├── training-handoff (single long-running task)
    │   Generate training script + Watchdog prompt + experiment context
    │   |
    │   watchdog (independent session)
    │   Active monitoring: auto-restart, parameter fixing, anomaly diagnosis
    │
    └── autoresearch-handoff (automated iteration)
        Generate research protocol + startup prompt + experience log
        |
        autoresearch-run → autoresearch (independent session, Supervisor)
        Each round: dispatch fresh Researcher subagent (design + code) →
        compliance check → Supervisor trains → runs fixed eval script →
        commit on improvement / rollback on regression → accumulate
        experience. Human stays on the loop, not in it.
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

Problems are classified into 3 tiers: environment problems (restart), simple parameter problems (fix + restart), and complex problems (sub-agent or report). When training completes, the user starts a new session on the experiment directory to analyze results.

### Auto Research

Auto Research turns a VP-validated baseline into an autonomous iteration loop. Instead of the human running each trial, a **Supervisor** session drives many rounds against a frozen protocol and surfaces results for the human to steer from above.

**Human on the Loop, not in it.** The human sees each round's result in the Task List, injects guidance by editing the protocol or adding notes, and reviews history via `experiences.md` and git — but does not approve individual rounds. The Supervisor keeps the loop running until the target is hit or `max_rounds` is exhausted.

**Protocol-driven.** `autoresearch-protocol.md` defines the search space before the loop starts:

| Field | Purpose |
|-------|---------|
| `research_question`, `target`, `baseline` | What to improve and by how much |
| `Fixed.files` | Code the Researcher may NOT modify (training loop, eval, data) |
| `Variable.files` + `Variable.range` | Code the Researcher may modify (e.g., model.py, hyperparams) |
| `Fixed.time_limit` / `epoch_limit` | Per-round training budget |
| `Eval.command` / `metric` / `direction` | The one and only source of truth for improvement |
| `max_rounds` | Hard stop |

**Each round, the Supervisor runs 7 steps (S1–S6 plus reporting):**

1. **Dispatch Researcher** — fresh subagent, background. Receives constraints + the last N rows of `experiences.md` (not the full history). Designs a strategy, writes code to `Variable.files`, reports "Code ready".
2. **Compliance check** — `git diff --name-only`. Any `Fixed.files` touched → round fails, rollback.
3. **Train** — Supervisor runs `train_command` within `time_limit`.
4. **Evaluation** — Supervisor runs the **fixed** `eval_command`. Training log metrics do not count.
5. **Act on result** — improved → `git commit`; not improved → `git checkout -- . && git clean -fd`. `experiences.md` is preserved across rollback so insights survive.
6. **Termination check** — target reached or `max_rounds` hit.
7. **Report progress** — update Task List, continue.

**Safety properties:**

- **Supervisor owns all git writes** — Researcher has file and bash permissions but cannot commit, reset, or checkout. One entity controls history.
- **Programmatic eval, not agent-generated eval** — `eval_command` is fixed before the loop starts and lives in `Fixed.files`. Researchers cannot write alternative metric code, even in new files. This prevents self-deception where an agent could (intentionally or not) produce favorable numbers.
- **Fresh Researcher per round** — no cross-round agent memory. Experience transfer happens through files, not hidden context, so every lesson is auditable.
- **Speed-first baseline** — first step/epoch must print quickly. Slow baseline = slow loop, so speed is a precondition of entering Auto Research, not something to optimize later.
- **Scheduling safety net** — layered CronCreate reminders (per-round timeout, check-in reminder, 30-minute heartbeat) prevent the loop from stalling silently if a background task hangs.

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
| **autoresearch-create** | Explicit entry point that activates Auto Research mode and routes into protocol-driven brainstorming |
| **autoresearch-handoff** | After VP passes, verify base code, extract the research protocol, and produce the run prompt |
| **autoresearch-run** | Explicit entry point that locates the protocol and starts the autonomous iteration |
| **autoresearch** | Supervisor loop: dispatch Researcher subagents, run eval, manage git, accumulate experience |

### Validation Pyramid

| Skill | Checks |
|-------|--------|
| **validation-pyramid** | 2-level validation orchestration integrated into ml-subagent-dev workflow |
| **ml-static-checks** | L0: Static analysis — device consistency, precision, FA, optimizer, DataLoader, logging & observability + 15 advisory checks |
| **ml-runtime-validator** | L1: Runtime validation — train ~5 min with metrics, then verify full pipeline (checkpoint, inference, evaluation) |

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
