# VP Logging & Observability Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add training log output validation checks to L0 (static) and L1 (runtime) of the Validation Pyramid, and rename `ml-code-reviewer` to `vp-static-checks`.

**Architecture:** Extend existing L0 and L1 skill files with new check items. Rename the L0 skill/agent/directory from `ml-code-reviewer` to `vp-static-checks`. Add a brainstorming question for visualization tool preference. Propagate the rename across all referencing files.

**Tech Stack:** Markdown skill definitions (no code changes)

**Spec:** `docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md`

---

### Task 1: Rename L0 skill directory

**Files:**
- Rename: `skills/ml-code-reviewer/` → `skills/vp-static-checks/`

- [ ] **Step 1: Rename the directory**

```bash
git mv skills/ml-code-reviewer skills/vp-static-checks
```

- [ ] **Step 2: Verify rename**

```bash
ls skills/vp-static-checks/
```
Expected: `SKILL.md  checklist.md`

- [ ] **Step 3: Commit**

```bash
git add skills/vp-static-checks/ && git commit -m "refactor: rename ml-code-reviewer directory to vp-static-checks"
```

---

### Task 2: Update L0 SKILL.md for new name and scope

**Files:**
- Modify: `skills/vp-static-checks/SKILL.md`

- [ ] **Step 1: Update frontmatter name**

Change `name: ml-code-reviewer` to `name: vp-static-checks`.

- [ ] **Step 2: Update title and description**

Change title from `# L0: ML Static Analysis (ML Code Reviewer)` to `# L0: VP Static Checks`.

Update Overview paragraph: replace "A specialized code-reviewer subagent that checks ML-specific static correctness" with "A specialized static-analysis subagent that checks ML code correctness and training observability."

- [ ] **Step 3: Update severity tiers**

Change:
```markdown
- **Mandatory checks (1-6):** Failure is Critical — blocks progress, Implementer must fix
- **Advisory checks (7-18):** Failure is Warning — reported but does not block progress
```
To:
```markdown
- **Mandatory checks (1-6, 19-20, 24):** Failure is Critical — blocks progress, Implementer must fix
- **Advisory checks (7-18, 21-23):** Failure is Warning — reported but does not block progress
```

- [ ] **Step 4: Update How It Works**

In the numbered list under "How It Works", change step 1: `Orchestrator dispatches `ml-code-reviewer` agent (defined in `agents/ml-code-reviewer.md`)` → `Orchestrator dispatches `vp-static-checks` agent (defined in `agents/vp-static-checks.md`)`.

- [ ] **Step 5: Update Integration section**

Change `**spml:subagent-dev** — dispatches this as a review stage` to keep as-is (no name reference).
Verify all self-references use `vp-static-checks`.

- [ ] **Step 6: Verify the file reads correctly**

Read the full file and confirm all references to `ml-code-reviewer` are replaced.

- [ ] **Step 7: Commit**

```bash
git add skills/vp-static-checks/SKILL.md && git commit -m "refactor: update L0 SKILL.md for vp-static-checks rename and expanded scope"
```

---

### Task 3: Add logging checks 19-24 to L0 checklist

**Files:**
- Modify: `skills/vp-static-checks/checklist.md`

- [ ] **Step 1: Add mandatory checks 19, 20, 24 to the Mandatory table**

After row 6 (DataLoader config) in the Mandatory table, add:

```markdown
| 19 | Loss file output | Always | Code writes loss values to a file (not only stdout) — file-writing patterns associated with loss |
| 20 | Step speed / throughput file output | Always | Code writes step time or throughput to a file |
| 24 | Visualization tool correctness | Enabled in experiment design doc | Selected tool has init + log calls + frequency control; skip if not enabled |
```

- [ ] **Step 2: Update Mandatory section header**

Change `## Mandatory (Critical) — checks 1-6` to `## Mandatory (Critical) — checks 1-6, 19-20, 24`.

- [ ] **Step 3: Add advisory checks 21, 22, 23 to the Advisory table**

After row 18 (CUDA kernel selection) in the Advisory table, add:

```markdown
| 21 | Data loading duration log | Has DataLoader | Code records data loading start/end/duration |
| 22 | Output frequency control | Has file logging | Log output has interval control (e.g., `if step % N == 0`, time-based gating); not every step |
| 23 | Progress bar | Always | Code uses a progress bar library (tqdm, rich.progress, etc.); tool not restricted |
```

- [ ] **Step 4: Update Advisory section header**

Change `## Advisory (Warning) — checks 7-18` to `## Advisory (Warning) — checks 7-18, 21-23`.

- [ ] **Step 5: Update "Adding New Checks" section**

Change `Update the agent definition in `agents/ml-code-reviewer.md` to match` to `Update the agent definition in `agents/vp-static-checks.md` to match`.

- [ ] **Step 6: Verify the complete checklist**

Read the full file. Confirm:
- Mandatory table has rows 1-6, 19, 20, 24
- Advisory table has rows 7-18, 21, 22, 23
- No duplicate numbers
- All conditions and descriptions match the spec

- [ ] **Step 7: Commit**

```bash
git add skills/vp-static-checks/checklist.md && git commit -m "feat: add logging & observability checks 19-24 to L0 checklist"
```

---

### Task 4: Rename and update agent definition

**Files:**
- Rename: `agents/ml-code-reviewer.md` → `agents/vp-static-checks.md`
- Modify: `agents/vp-static-checks.md`

- [ ] **Step 1: Rename the agent file**

```bash
git mv agents/ml-code-reviewer.md agents/vp-static-checks.md
```

- [ ] **Step 2: Update frontmatter name and description**

Change `name: ml-code-reviewer` to `name: vp-static-checks`.

Change description from `ML-specialized code reviewer that checks static correctness of ML code — device consistency, precision, FlashAttention, optimizer coverage, and 12 additional advisory checks.` to `Static analysis agent that checks ML code correctness and training observability — device consistency, precision, FlashAttention, optimizer coverage, logging & observability, and 15 additional advisory checks.`

- [ ] **Step 3: Update title**

Change `# ML Code Reviewer` to `# VP Static Checks`.

- [ ] **Step 4: Add checks 19-24 to Section 1 tables**

In the **Mandatory (Critical)** table, after row 6, add:

```markdown
| 19 | Loss file output | Always | Code writes loss values to a file (not only stdout) |
| 20 | Step speed / throughput file output | Always | Code writes step time or throughput to a file |
| 24 | Visualization tool correctness | Enabled in experiment design doc | Selected tool has init + log calls + frequency control; skip if not enabled |
```

In the **Advisory (Warning)** table, after row 18, add:

```markdown
| 21 | Data loading duration log | Has DataLoader | Code records data loading start/end/duration |
| 22 | Output frequency control | Has file logging | Log output has interval control; not triggered every step |
| 23 | Progress bar | Always | Code uses a progress bar library (tqdm, rich.progress, etc.) |
```

- [ ] **Step 5: Verify the output format template**

The ML Static Analysis Results template uses generic `Check #N` — no number range to update. Verify it still reads correctly with 24 checks.

- [ ] **Step 6: Verify the file**

Read the full file. Confirm all `ml-code-reviewer` references are gone and new checks are present.

- [ ] **Step 7: Commit**

```bash
git add agents/ && git commit -m "refactor: rename ml-code-reviewer agent to vp-static-checks and add logging checks"
```

---

### Task 5: Add Logging Output Validation section to L1

**Files:**
- Modify: `skills/ml-runtime-validator/SKILL.md`

- [ ] **Step 1: Add new section after Architecture-Specific Checks**

After the Architecture-Specific Checks table (line 48), add:

```markdown
### Logging Output Validation

Checks that the training code's logging actually produces correct output at runtime. Each check validates three layers: **existence → frequency → value correctness**.

| # | Check | Severity | Validation Method |
|---|-------|----------|-------------------|
| L.1 | Loss file output correctness | **Mandatory** | File exists, non-empty, parseable format; values reasonable (no all-NaN/Inf/zero, trend consistent with gradient behavior) |
| L.2 | Step speed output correctness | **Mandatory** | File exists, non-empty; values match wall clock (step count × reported step time ≈ actual elapsed time) |
| L.3 | Data loading duration correctness | Advisory | Duration record exists; values reasonable (non-zero, non-negative, consistent with actual time window) |
| L.4 | Output frequency reasonableness | Advisory | Actual log entry timestamps have intervals approximately minute-level (complements L0 check 22 which verifies interval-control logic in code) |
| L.5 | Progress bar correctness | Advisory | Progress bar total matches training target — 1 epoch → dataset size; N steps → total = N; T minutes → time-based estimate; advance rate matches actual speed |
| L.6 | Visualization tool output correctness (if enabled) | **Mandatory** | Output directory/API has data; frequency reasonable; values cross-validated against loss/speed files for consistency; skip if not enabled |
```

- [ ] **Step 2: Update Anomaly Detection section**

In the "Catches obvious problems" list, add:
```markdown
- Logging outputs missing or empty (L.1, L.2 checks)
```

- [ ] **Step 3: Update the Integration section references**

Change `**spml:ml-code-reviewer** — must pass before L1 runs` to `**spml:vp-static-checks** — must pass before L1 runs`.

- [ ] **Step 4: Update When to Use section**

Change `After L0 (spml:ml-code-reviewer) passes` to `After L0 (spml:vp-static-checks) passes`.

- [ ] **Step 5: Verify the file**

Read the full file. Confirm:
- New Logging Output Validation section present with 6 checks
- All `ml-code-reviewer` references replaced with `vp-static-checks`
- Existing sections unchanged

- [ ] **Step 6: Commit**

```bash
git add skills/ml-runtime-validator/SKILL.md && git commit -m "feat: add Logging Output Validation section to L1 runtime validator"
```

---

### Task 6: Add visualization tool question to brainstorming

**Files:**
- Modify: `skills/brainstorming/SKILL.md` (around line 91-111, "Confirming validation scope" section)

- [ ] **Step 1: Update L0 section in validation scope**

Change:
```markdown
**L0: ML Static Analysis (spml:ml-code-reviewer)**
- Always enabled for ML code tasks
- Checks: device consistency, precision, FA, optimizer, scheduler, DataLoader (mandatory); plus 12 advisory checks
- Ask: "Any project-specific checks to add?"
```
To:
```markdown
**L0: VP Static Checks (spml:vp-static-checks)**
- Always enabled for ML code tasks
- Checks: device consistency, precision, FA, optimizer, scheduler, DataLoader, loss/speed file output, visualization tool (mandatory); plus 15 advisory checks
- Ask: "Do you need visualization metrics output (e.g., WandB, TensorBoard, MLflow)? If yes, which tool?"
- Ask: "Any project-specific checks to add?"
```

- [ ] **Step 2: Verify the change**

Read lines 91-115 of the file. Confirm the visualization question is present and the L0 name is updated.

- [ ] **Step 3: Commit**

```bash
git add skills/brainstorming/SKILL.md && git commit -m "feat: add visualization tool question to brainstorming validation scope"
```

---

### Task 7: Propagate rename across all referencing files

**Files:**
- Modify: `skills/validation-pyramid/SKILL.md` (lines 31, 43, 49)
- Modify: `skills/subagent-dev/SKILL.md` (line 234)
- Modify: `skills/using-superpowers-ml/SKILL.md` (line 108)
- Modify: `skills/diagnostics/SKILL.md` (line 196)
- Modify: `README.md` (line 123)

- [ ] **Step 1: Update validation-pyramid/SKILL.md**

Replace all instances of `ml-code-reviewer` with `vp-static-checks`:
- Line 31: `├─ L0: ML Code Reviewer (spml:ml-code-reviewer)` → `├─ L0: VP Static Checks (spml:vp-static-checks)`
- Line 43: `| L0 | spml:ml-code-reviewer |` → `| L0 | spml:vp-static-checks |`
- Line 49: `the ml-code-reviewer agent` → `the vp-static-checks agent`

Also update the L0 description in the Level Summary table to mention logging/observability:
```
| L0 | spml:vp-static-checks | Static config errors, logging & observability (device, precision, optimizer, DataLoader, loss/speed output) | Seconds (code review) |
```

- [ ] **Step 2: Update subagent-dev/SKILL.md**

Replace all 6 occurrences of `ml-code-reviewer` / `ML Code Reviewer`:
- Line 14: `L0: ML Code Reviewer subagent` → `L0: VP Static Checks subagent`
- Line 45: `"L0: ML Code Reviewer"` (dot graph node) → `"L0: VP Static Checks"`
- Lines 71-72, 75: `"L0: ML Code Reviewer"` (dot graph edges) → `"L0: VP Static Checks"`
- Line 234: `**spml:ml-code-reviewer**` → `**spml:vp-static-checks**`

- [ ] **Step 3: Update using-superpowers-ml/SKILL.md**

At line 108, replace `ml-code-reviewer` with `vp-static-checks`.

- [ ] **Step 4: Update diagnostics/SKILL.md**

At line 196, replace `spml:ml-code-reviewer` with `spml:vp-static-checks`.

- [ ] **Step 5: Update README.md**

At line 123, replace `ml-code-reviewer` with `vp-static-checks` and update the description to include logging checks:
```markdown
| **vp-static-checks** | L0: Static analysis — device consistency, precision, FA, optimizer, DataLoader, logging & observability + 15 advisory checks |
```

- [ ] **Step 6: Verify no remaining references**

```bash
grep -r "ml-code-reviewer\|ML Code Reviewer" skills/ agents/ README.md
```
Expected: No output (zero matches). Historical docs in `docs/` are intentionally not updated.

- [ ] **Step 7: Commit**

```bash
git add skills/validation-pyramid/SKILL.md skills/subagent-dev/SKILL.md skills/using-superpowers-ml/SKILL.md skills/diagnostics/SKILL.md README.md && git commit -m "refactor: propagate ml-code-reviewer → vp-static-checks rename across all skills"
```

---

### Task 8: Sync plugin cache

**Files:**
- Modify: Plugin cache at `~/.claude/plugins/cache/spml-dev/spml/0.1.0/`

- [ ] **Step 1: Reinstall the plugin**

The simplest approach is to reinstall the plugin so the cache is rebuilt from source. Follow the project's plugin installation process.

If manual sync is needed:
```bash
# Remove old cache entries
rm -rf ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/ml-code-reviewer
rm -rf ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/vp-engineering-efficiency
rm -rf ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/vp-process-metrics
rm -rf ~/.claude/plugins/cache/spml-dev/spml/0.1.0/agents/ml-code-reviewer.md

# Copy updated source to cache
cp -r skills/vp-static-checks ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
cp -r skills/ml-runtime-validator ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
cp -r skills/brainstorming ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
cp -r skills/validation-pyramid ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
cp -r skills/subagent-dev ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
cp -r skills/using-superpowers-ml ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
cp -r skills/diagnostics ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
cp agents/vp-static-checks.md ~/.claude/plugins/cache/spml-dev/spml/0.1.0/agents/
```

- [ ] **Step 2: Verify cache is correct**

```bash
ls ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/vp-static-checks/
ls ~/.claude/plugins/cache/spml-dev/spml/0.1.0/agents/vp-static-checks.md
```
Expected: Both paths exist.

- [ ] **Step 3: Verify old cache entries are gone**

```bash
ls ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/ml-code-reviewer/ 2>/dev/null && echo "STALE" || echo "CLEAN"
ls ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/vp-engineering-efficiency/ 2>/dev/null && echo "STALE" || echo "CLEAN"
```
Expected: Both print `CLEAN`.

- [ ] **Step 4: Commit (if plugin.json changed)**

Only if the plugin registration file needed updating. Otherwise no commit needed for this task.

---

### Task 9: Update spec status

**Files:**
- Modify: `docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md`

- [ ] **Step 1: Update status**

Change `**Status:** Draft` to `**Status:** Implemented`.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-03-17-vp-logging-observability-design.md && git commit -m "docs: mark vp-logging-observability spec as implemented"
```
