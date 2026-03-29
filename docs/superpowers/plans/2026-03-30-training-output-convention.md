# Training Output Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update VP checks (L0/L1) and training-handoff docs to enforce the two-layer output convention (console for humans, files for agents) plus checkpoint configurability.

**Architecture:** Pure documentation/skill-file edits across 4 files. No code changes. Each task modifies one file independently.

**Tech Stack:** Markdown skill files

**Spec:** `docs/superpowers/specs/2026-03-30-training-output-convention-design.md`

---

### Task 1: Update L0 checklist — checks #22, #23, add #25

**Files:**
- Modify: `skills/ml-static-checks/checklist.md:42-43` (Advisory table rows for #22 and #23)
- Modify: `skills/ml-static-checks/checklist.md:44` (add new row #25 before "Adding New Checks" section)

- [ ] **Step 1: Edit check #22 in Advisory table**

Replace the existing row:

```
| 22 | Output frequency control | Has file logging | Log output has interval control (e.g., `if step % N == 0`, time-based gating); not every step |
```

With:

```
| 22 | Output frequency control | Has file logging or console output | **Console**: tqdm (preferred) or print has minute-level frequency control — tqdm via `mininterval`, print via `if step % N == 0` with time-based gating. **File**: log output has interval control; not every step |
```

- [ ] **Step 2: Edit check #23 in Advisory table**

Replace the existing row:

```
| 23 | Progress bar | Always | Code uses a progress bar library (tqdm, rich.progress, etc.); tool not restricted |
```

With:

```
| 23 | Console metrics display | Always | Console output (tqdm or print) carries key runtime metrics (at least loss) — via tqdm `set_postfix` or formatted print string |
```

- [ ] **Step 3: Add check #25 to Advisory table**

Add the following row after check #23:

```
| 25 | Checkpoint interval configurability | Has checkpoint saving | Checkpoint save interval is configurable (via argument/config, not hardcoded) and default value is reasonable |
```

- [ ] **Step 4: Verify the checklist reads correctly**

Read `skills/ml-static-checks/checklist.md` and confirm:
- #22 now covers both console and file frequency
- #23 now requires key metrics in console output
- #25 exists in Advisory table with correct condition
- Mandatory table header still lists `19-20, 24` (no change)
- Advisory table header lists `7-18, 21-23, 25`

- [ ] **Step 5: Commit**

```bash
git add skills/ml-static-checks/checklist.md
git commit -m "feat(vp): update L0 checks #22, #23 and add #25 for output convention"
```

---

### Task 2: Sync L0 agent definition — checks #22, #23, add #25

**Files:**
- Modify: `agents/ml-static-checks.md:50-51` (Advisory table rows for #22 and #23)
- Modify: `agents/ml-static-checks.md:52` (add new row #25)

- [ ] **Step 1: Edit check #22 in Advisory table**

Replace the existing row:

```
| 22 | Output frequency control | Has file logging | Log output has interval control; not triggered every step |
```

With:

```
| 22 | Output frequency control | Has file logging or console output | **Console**: tqdm (preferred) or print has minute-level frequency control. **File**: log output has interval control; not every step |
```

- [ ] **Step 2: Edit check #23 in Advisory table**

Replace the existing row:

```
| 23 | Progress bar | Always | Code uses a progress bar library (tqdm, rich.progress, etc.) |
```

With:

```
| 23 | Console metrics display | Always | Console output (tqdm or print) carries key runtime metrics (at least loss) — via tqdm `set_postfix` or formatted print string |
```

- [ ] **Step 3: Add check #25 to Advisory table**

Add the following row after check #23:

```
| 25 | Checkpoint interval configurability | Has checkpoint saving | Checkpoint save interval is configurable (via argument/config, not hardcoded) and default value is reasonable |
```

- [ ] **Step 4: Verify consistency with checklist.md**

Read `agents/ml-static-checks.md` and confirm checks #22, #23, #25 match the wording in `skills/ml-static-checks/checklist.md` from Task 1 (same semantics, may be slightly condensed for the agent table).

- [ ] **Step 5: Commit**

```bash
git add agents/ml-static-checks.md
git commit -m "feat(vp): sync L0 agent definition with updated checks #22, #23, #25"
```

---

### Task 3: Update L1 runtime validator — L.4, L.5, add L.7

**Files:**
- Modify: `skills/ml-runtime-validator/SKILL.md:96-97` (Logging Output Validation table rows L.4 and L.5)
- Modify: `skills/ml-runtime-validator/SKILL.md:98` (add new row L.7 after L.6)

- [ ] **Step 1: Edit L.4 in Logging Output Validation table**

Replace the existing row:

```
| L.4 | Output frequency reasonableness | Advisory | Actual log entry timestamps have intervals approximately minute-level (complements L0 check 22 which verifies interval-control logic in code) |
```

With:

```
| L.4 | Output frequency reasonableness | Advisory | **Console**: tqdm refresh interval or print output interval is approximately minute-level. **File**: actual log entry timestamps have intervals approximately minute-level. Complements L0 check 22 |
```

- [ ] **Step 2: Edit L.5 in Logging Output Validation table**

Replace the existing row:

```
| L.5 | Progress bar correctness | Advisory | Progress bar total matches training target — 1 epoch → dataset size; N steps → total = N; T minutes → time-based estimate; advance rate matches actual speed |
```

With:

```
| L.5 | Console metrics correctness | Advisory | Progress bar total matches training target; advance rate matches actual speed. Console output (tqdm postfix or print) includes key runtime metrics (at least loss); metric values are consistent with those recorded in file logs |
```

- [ ] **Step 3: Add L.7 to Logging Output Validation table**

Add the following row after L.6:

```
| L.7 | Checkpoint periodic saving | Advisory | During L1 training (~5 min), at least one checkpoint file is produced (if configured interval should trigger within 5 min); checkpoint file is non-empty and loadable (reuse Stage 4 verification logic) |
```

- [ ] **Step 4: Verify the table reads correctly**

Read `skills/ml-runtime-validator/SKILL.md` lines 88-100 and confirm:
- L.4 now covers both console and file frequency
- L.5 now requires metrics consistency between console and file
- L.7 exists with checkpoint verification description
- L.1, L.2, L.3, L.6 are unchanged

- [ ] **Step 5: Commit**

```bash
git add skills/ml-runtime-validator/SKILL.md
git commit -m "feat(vp): update L1 checks L.4, L.5 and add L.7 for output convention"
```

---

### Task 4: Update training-handoff — upstream requirements and Step 3 checklist

**Files:**
- Modify: `skills/training-handoff/SKILL.md:86-93` (Upstream: Production Script Requirements bullet list)
- Modify: `skills/training-handoff/SKILL.md:58-62` (Step 3 Expected checklist)

- [ ] **Step 1: Edit Upstream: Production Script Requirements**

Replace these lines (under the "Key requirements to include in plans:" paragraph):

```markdown
- Human-readable log file output (one line per step with key metrics)
- MFU calculation and logging
- Terminal progress bar (tqdm)
- Checkpoint save/resume support
```

With:

```markdown
- Console output: tqdm (preferred) or controlled print, minute-level frequency, carrying key metrics (loss, lr, etc.)
- File output: detailed metrics written to file (loss, grad_norm, lr, mfu, memory, step_time), for Agent exploration
- Checkpoint: periodic save with configurable interval, resume support
```

- [ ] **Step 2: Edit Step 3 Expected checklist**

Replace these lines in the Expected checklist:

```markdown
- [ ] Terminal progress indicator (tqdm or similar)
- [ ] Key metrics in log: loss, gradient norm, learning rate
- [ ] MFU in log (needed for efficiency monitoring)
- [ ] Checkpoint support with configurable interval
```

With:

```markdown
- [ ] Console output uses tqdm (preferred) or controlled print, minute-level, carrying key metrics
- [ ] Detailed metrics written to file: loss, gradient norm, learning rate, step time
- [ ] MFU in file log (needed for efficiency monitoring)
- [ ] Checkpoint save with configurable interval
```

- [ ] **Step 3: Verify the file reads correctly**

Read `skills/training-handoff/SKILL.md` and confirm:
- Upstream section has three-layer bullet points (console, file, checkpoint)
- Step 3 Expected checklist has the updated items
- All other content (Required checklist, evaluation checks, etc.) is unchanged

- [ ] **Step 4: Commit**

```bash
git add skills/training-handoff/SKILL.md
git commit -m "feat(handoff): update production script requirements for output convention"
```

---

### Task 5: Final cross-file verification

**Files:**
- Read: all 4 modified files

- [ ] **Step 1: Cross-check consistency**

Read all 4 files and verify:
- `checklist.md` and `agents/ml-static-checks.md` have identical semantics for #22, #23, #25
- L1 checks L.4, L.5, L.7 complement L0 checks #22, #23, #25 respectively
- training-handoff upstream requirements align with what L0/L1 will check

- [ ] **Step 2: Version bump**

Bump patch version in the project's version file (this is a non-breaking enhancement to existing checks).

- [ ] **Step 3: Final commit**

```bash
git add <version-file>
git commit -m "chore: bump version for training output convention"
```
