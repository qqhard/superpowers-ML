# Training Output Convention Design

**Date:** 2026-03-30
**Status:** Draft
**Scope:** L0/L1 check modifications + training-handoff doc update

## Problem

Training scripts have three audiences: human users watching the console, Agent (Watchdog) exploring file logs, and downstream evaluation pipelines consuming checkpoints. The current VP checks and skill docs don't clearly separate these output layers, leading to ambiguous requirements (e.g., "one line per step" conflates console and file output).

## Design Principle: Two-Layer Output + Checkpoint

| Layer | Target Audience | Medium | Frequency | Content |
|-------|----------------|--------|-----------|---------|
| **Console** | Human user | tqdm (preferred) or controlled print | ~minute-level | Key indicators: loss, lr, step, etc. Human sees the program is running normally |
| **File** | Agent (Watchdog) | Log files, wandb, etc. | Interval-controlled (not every step) | Detailed metrics: loss, grad_norm, lr, mfu, memory, step_time, etc. Agent explores via tools to avoid excessive token consumption |
| **Checkpoint** | Evaluation pipeline | Model/optimizer state files | User-configurable interval | Periodic saves; Watchdog can spawn async subagent for evaluation without interrupting training |

Key relationships:
- Console and file are OR layers — both required, serve different audiences
- tqdm and print are OR for console — tqdm preferred, print as fallback
- Agent reads file logs by exploring (tail, grep), not by ingesting the entire file

## Changes

### L0 Static Checks

#### Modified: #22 Output frequency control (Advisory)

**Before:**
- Condition: `Has file logging`
- Verify: Log output has interval control (e.g., `if step % N == 0`, time-based gating); not every step

**After:**
- Condition: `Has file logging or console output`
- Verify:
  - **Console**: tqdm (preferred) or print output has minute-level frequency control — tqdm via `mininterval` or similar, print via `if step % N == 0` with time-based gating
  - **File**: Log output has interval control; not every step (unchanged)

#### Modified: #23 Progress bar (Advisory)

**Before:**
- Condition: `Always`
- Verify: Code uses a progress bar library (tqdm, rich.progress, etc.); tool not restricted

**After:**
- Condition: `Always`
- Verify: Console output (tqdm or print) carries key runtime metrics (at least loss) — via tqdm `set_postfix` or formatted print string

#### New: #25 Checkpoint interval configurability (Advisory)

- Condition: `Has checkpoint saving`
- Verify: Checkpoint save interval is configurable (via argument/config, not hardcoded) and default value is reasonable

### L1 Runtime Validation

#### Modified: L.4 Output frequency reasonableness (Advisory)

**Before:**
- Actual log entry timestamps have intervals approximately minute-level

**After:**
- **Console**: tqdm refresh interval or print output interval is approximately minute-level
- **File**: Actual log entry timestamps have intervals approximately minute-level (unchanged)

#### Modified: L.5 Progress bar correctness (Advisory)

**Before:**
- Progress bar total matches training target; advance rate matches actual speed

**After:**
- Progress bar total matches training target; advance rate matches actual speed (unchanged)
- Console output (tqdm postfix or print) includes key runtime metrics (at least loss); metric values are consistent with those recorded in file logs

#### New: L.7 Checkpoint periodic saving (Advisory)

- During L1 training (~5 min), at least one checkpoint file is produced (if configured interval should trigger within 5 min)
- Checkpoint file is non-empty and loadable (reuse Stage 4 checkpoint verification logic)

### training-handoff SKILL.md

#### Modified: "Upstream: Production Script Requirements" section

**Before:**
```
- Human-readable log file output (one line per step with key metrics)
- MFU calculation and logging
- Terminal progress bar (tqdm)
- Checkpoint save/resume support
```

**After:**
```
- Console output: tqdm (preferred) or controlled print, minute-level frequency, carrying key metrics (loss, lr, etc.)
- File output: detailed metrics written to file (loss, grad_norm, lr, mfu, memory, step_time), for Agent exploration
- Checkpoint: periodic save with configurable interval, resume support
```

#### Modified: Step 3 Expected checklist

**Before:**
```
- [ ] Terminal progress indicator (tqdm or similar)
- [ ] Key metrics in log: loss, gradient norm, learning rate
- [ ] MFU in log (needed for efficiency monitoring)
- [ ] Checkpoint support with configurable interval
```

**After:**
```
- [ ] Console output uses tqdm (preferred) or controlled print, minute-level, carrying key metrics
- [ ] Detailed metrics written to file: loss, gradient norm, learning rate, step time
- [ ] MFU in file log (needed for efficiency monitoring)
- [ ] Checkpoint save with configurable interval
```

### agents/ml-static-checks.md

Sync checks #22, #23, and add #25 to match the checklist changes above.

### skills/ml-static-checks/checklist.md

Sync checks #22, #23, and add #25 to match the changes above.

## Out of Scope

- **Watchdog async evaluation via subagent** — separate design
- **Specific file log format** (JSON, CSV, key=value) — remains flexible per current Watchdog design
- **wandb/TensorBoard integration details** — covered by existing check #24
