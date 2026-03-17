# Watchdog Active Shepherd Design

## Problem

The current Watchdog skill is read-only: it observes training metrics, diagnoses anomalies, and writes prompt files for a separate training-resume session to act on. This means every problem — even a transient environment crash — requires human intervention to restart training.

For long-running ML tasks (hours to days), this is impractical. Environment instability, OOM events, and simple parameter issues should be handled automatically. The human should only be involved when the problem is genuinely complex.

## Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Directly transform Watchdog from read-only observer to active shepherd | Simpler than maintaining separate observer + actor skills |
| Intervention model | Three-tier classification with mode-controlled execution | Matches natural problem severity hierarchy |
| Restart mechanism | Direct `bash` execution of training script | Training scripts already support checkpoint resume; no indirection needed |
| Complex problem handling (Autonomous mode) | Spawn sub-agent running training-resume flow | Protects Watchdog's context window from complex debugging |
| Environment retry limit | No limit, but notify user after repeated crashes | Environment problems are environment problems; keep retrying, but keep human informed |
| Mode selection timing | Configurable at training-handoff or Watchdog startup (startup can override) | Flexibility without complexity |
| VP after auto-fix | Not required | VP already validated code during initial development; runtime fixes are incremental |
| Log format | Human-readable text (not JSONL) | LLM Watchdog parses text natively; humans can read logs directly; simpler training scripts |
| Execution mode | Always combined (Watchdog can launch/restart training) | Active intervention requires process control; separated mode is incompatible with Guardian/Autonomous |

## Design

### 1. Operating Modes

Three modes controlling Watchdog's intervention authority:

| Mode | Environment crash | Simple parameter issue | Complex issue |
|------|------------------|----------------------|---------------|
| **Monitor** | Report only | Report only | Report only |
| **Guardian** (default) | Auto-restart | Auto-fix + restart | Report to user |
| **Autonomous** | Auto-restart | Auto-fix + restart | Spawn sub-agent to fix + restart |

Mode is stored in `experiment-context.md` as `watchdog_mode: monitor | guardian | autonomous`. Set during training-handoff, overridable at Watchdog startup.

### 2. Problem Classification

Watchdog is an LLM — it uses judgment, not rigid rules. The following are guiding examples, not exhaustive lists.

**Tier 1: Environment problems** (process died from external causes)

- OOM killer (exit code 137)
- NCCL timeout / network errors
- Hardware GPU errors (not caused by code)
- Disk full
- SIGKILL / SIGTERM from external source

Action: Restart training script from latest checkpoint. No code changes. No retry limit — keep restarting. After repeated crashes (e.g., 5+ in a short period), notify the user that environment instability is persisting but continue retrying.

**Tier 2: Simple parameter problems** (needs numeric/config adjustment, no logic change)

- CUDA OOM → reduce batch size or increase gradient accumulation steps
- Loss explosion → lower learning rate or add gradient clipping
- Extended plateau → adjust lr schedule
- Any numeric parameter that is clearly misconfigured

Action: Modify parameter value in training script or config, record change in experiment-context.md, restart from checkpoint.

**Tier 3: Complex problems** (requires logic changes or root cause is unclear)

- Everything not in Tier 1 or 2
- Data issues (NaN in inputs, data loading errors)
- Model architecture problems (attention collapse, expert collapse)
- Code bugs

Action: Depends on mode — Guardian reports to user; Autonomous spawns sub-agent.

**Classification principle:** When uncertain, escalate upward (treat simple as complex). Never downgrade ambiguous problems.

### 3. Monitoring Loop

```
Startup → Read experiment-context.md → Confirm mode → Locate log file and training script
  │
  ▼
┌──────────────── Monitoring Loop ────────────────┐
│                                                  │
│  sleep(interval) → Read log tail                 │
│       │                                          │
│       ├─ New lines present → Parse metrics       │
│       │    │                                     │
│       │    ├─ NORMAL → Update step baseline       │
│       │    │           → Continue loop            │
│       │    │                                     │
│       │    ├─ ANOMALY → Classify problem          │
│       │    │    ├─ Tier 1 → Wait for exit, restart│
│       │    │    ├─ Tier 2 → Fix param + restart   │
│       │    │    └─ Tier 3 → Mode-dependent action │
│       │    │         ├─ Monitor/Guardian → Report │
│       │    │         └─ Autonomous → Spawn agent  │
│       │    │                                     │
│       │    └─ COMPLETE → Exit loop                │
│       │                                          │
│       └─ No new lines → Check process alive       │
│            ├─ Dead → Read exit code → Classify    │
│            └─ Alive → Assess hang                 │
│                 ├─ Within baseline → Wait more    │
│                 └─ Exceeds baseline significantly │
│                      → Kill process → Classify    │
│                                                  │
│  All interventions → Record in experiment-context │
│  After restart → Enter intensive observation      │
└──────────────────────────────────────────────────┘
  │
  ▼
Training complete → Write completion-prompt.md → Notify user
```

### 4. Polling and Hang Detection

**Log as heartbeat:** Training scripts output one line per step/epoch with key metrics. Each new log line is a heartbeat signal. The format is human-readable text — no JSONL required. Example:

```
step=100 loss=0.823 lr=0.0001 grad_norm=1.42 mfu=0.45 mem_mb=24531
step=101 loss=0.819 lr=0.0001 grad_norm=1.38 mfu=0.44 mem_mb=24528
```

The exact format is flexible (key=value, tabular, or any readable layout). Training-handoff specifies which metrics to include based on what VP L1 validated. Watchdog, as an LLM, parses any consistent text format. Training scripts should also print to terminal (tqdm or similar) for human monitoring.

**Polling interval:**
- Normal: 2–5 minutes (sampling, not every step)
- Post-anomaly / post-restart: 1 minute for 5 cycles, then back to normal
- Watchdog must use Bash tool `sleep` to implement intervals, ensuring it runs continuously and does not stall

**Hang detection:** Watchdog observes step intervals during normal monitoring and builds a baseline. If no new log line appears for significantly longer than the baseline (e.g., 10x the typical step duration), and the process is still alive, Watchdog judges the process as hung. It then kills the process and classifies the problem:
- Likely environment (deadlock, NCCL hang) → Tier 1, restart
- Possibly code issue → Escalate to Tier 2 or Tier 3

### 5. Restart Mechanism

Watchdog obtains the training script path and launch arguments from experiment-context.md (written by training-handoff). Restart = re-run the same command via `bash`. The training script's built-in checkpoint resume handles continuation from the latest saved state.

After restart, Watchdog enters **intensive observation** (1-minute interval, 5 cycles) to confirm training resumes normally.

### 6. Simple Parameter Modification

When Watchdog identifies a Tier 2 problem:

1. Record current parameter value in experiment-context.md (before)
2. Modify the parameter in the training script or config file
3. Record new value and rationale in experiment-context.md (after)
4. Restart from checkpoint

### 7. Sub-Agent Spawn (Autonomous Mode)

When Watchdog identifies a Tier 3 problem in Autonomous mode:

1. Write diagnosis to experiment-context.md
2. Generate recovery-prompt.md (same format as current design)
3. Spawn sub-agent using Claude Code's Agent tool with instructions: read recovery-prompt.md, follow training-resume flow, fix issue, restart training
4. Sub-agent does NOT run VP — trusts that initial VP validation covers code correctness
5. Watchdog waits for the Agent tool call to return, then resumes monitoring loop

Note: This introduces a Claude Code dependency for Autonomous mode. Monitor and Guardian modes remain framework-agnostic (only require bash access).

### 8. Logging and Audit Trail

All interventions are appended to experiment-context.md with:
- Timestamp
- Problem classification (Tier 1/2/3)
- Anomaly description (what was observed)
- Action taken
- Changes made (if any, with before/after values)

This provides a complete history for the user to review and for training-resume to reference in multi-round scenarios.

## Affected Skills

| Skill | Change scope |
|-------|-------------|
| `skills/watchdog/SKILL.md` | **Rewrite** — Remove read-only constraint, add three-tier classification, execution actions, mode control, human-readable log format. Remove separated/combined execution mode distinction (now always combined). |
| `skills/training-handoff/SKILL.md` | **Significant rewrite** — Change log requirements from JSONL to human-readable; add `watchdog_mode` field to experiment-context.md template; rewrite embedded watchdog-prompt.md template (currently says "DO NOT modify code, adjust hyperparameters" which contradicts new design). Remove separated execution mode. |
| `skills/training-resume/SKILL.md` | **Modify** — Update `metrics.jsonl` references to human-readable log file; add behavior for when spawned by Watchdog sub-agent (no VP, fix and restart) |

### Unaffected skills

- `skills/diagnostics/SKILL.md` — Diagnosis logic unchanged; training-resume invokes as needed
- VP skills — Not directly affected by this spec
- `skills/brainstorming/SKILL.md` — Not affected

### Related: VP logging & observability spec

The VP logging spec (`2026-03-17-vp-logging-observability-design.md`) defines L0 and L1 checks that reference structured log formats. Those checks need updating to validate human-readable log format. This is a follow-up task — the VP logging spec changes are out of scope for this spec but should be addressed before or during implementation.

## Execution Mode

The original Watchdog design offered "separated" (training runs independently, Watchdog only monitors) and "combined" (Watchdog launches training) modes. The new design **always operates in combined mode** — Watchdog must be able to launch and restart the training process. The separated/combined distinction is replaced by the monitor/guardian/autonomous mode system.

Monitor mode preserves the _observation-only behavior_ of the original design (report but don't intervene), but still assumes Watchdog has access to the training process for liveness checks.
