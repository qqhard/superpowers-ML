# Watchdog Sleep Mechanism Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Watchdog skill's monitoring loop actually execute by adding a hard gate for `bash sleep` and rewriting the loop as implementation-level instructions.

**Architecture:** Single-file edit to `skills/watchdog/SKILL.md`. Three changes: insert new hard gate, replace monitoring loop, delete redundant section.

**Tech Stack:** Markdown (skill definition)

---

### Task 1: Add Sleep Loop Hard Gate

**Files:**
- Modify: `skills/watchdog/SKILL.md:12-14` (insert after "Overview" section, before "When to Use" — matching the hard gate placement convention used by all other skills)

- [ ] **Step 1: Insert the hard gate block**

After the Overview section (line 12, `...from reporting only (Monitor) to fully autonomous recovery (Autonomous).`) and before `## When to Use` (line 14), insert:

```markdown

<HARD-GATE>
## Monitoring Loop Mechanism

You MUST use the Bash tool to execute `sleep <seconds>` as the ONLY way to wait between checks.

After each sleep returns, immediately proceed to the next check — do NOT wait for user input.

**Prohibited:**
- Outputting "I'll check in N minutes" and then stopping
- Writing a standalone monitoring/watchdog script that runs its own loop or scheduling (Python, shell, or any other). Inline one-liners for log parsing are fine.
- Using /loop or any external scheduling mechanism
- Asking the user to remind you to check

**Required execution pattern:**
1. Bash tool: `sleep <interval>`  (see Monitoring Loop step 1 for interval values)
2. Bash tool: `tail -20 <log_file>`  (read latest log lines)
3. Analyze output, report status
4. If anomaly → diagnose and act per operating mode
5. Go to step 1
</HARD-GATE>

```

- [ ] **Step 2: Verify structure**

Read the file and confirm the hard gate sits between "Overview" and "When to Use", matching the placement convention of other skills (diagnostics, training-handoff, verification, brainstorming).

- [ ] **Step 3: Commit**

```bash
git add skills/watchdog/SKILL.md
git commit -m "feat(watchdog): add hard gate for sleep-based monitoring loop"
```

---

### Task 2: Rewrite Monitoring Loop

**Files:**
- Modify: `skills/watchdog/SKILL.md` — replace "Monitoring Loop" section (current lines 102-127)

- [ ] **Step 1: Replace the monitoring loop**

Replace the entire `## Monitoring Loop` section (from `## Monitoring Loop` through the closing ` ``` `) with:

````markdown
## Monitoring Loop

```
loop {
    1. Bash tool: `sleep <interval_seconds>`
       - Normal: 120-300s (2-5 min)
       - Post-restart / post-anomaly: 60s for 5 cycles, then back to normal
    2. Bash tool: `tail -20 <log_file>` (each new log line = heartbeat; format is human-readable text, not JSONL)
    3. Check for new lines since last check:
       a. New lines → parse metrics, go to step 4
       b. No new lines → Bash tool: `ps aux | grep <training_script>`
          - Process dead → read exit code → classify problem tier → act
          - Process alive → compare silence duration vs step baseline
            - Startup grace period (first 15 min or until 3 logged steps): do not classify silence as hang
            - Within baseline → continue (go to step 1)
            - Exceeds 10x baseline → kill process → classify problem tier (environment hang → Tier 1; possible code issue → Tier 2/3)
    4. Analyze metrics:
       a. Sanity: NaN, Inf, negative loss, zero gradient
       b. Baseline comparison vs VP ranges in experiment-context.md
       c. Trend: loss decreasing? grad_norm stable?
       d. Anomaly patterns: spike, plateau, divergence, sudden shift
    5. Classify:
       - NORMAL → update step time baseline, output one-line progress, go to step 1
       - ANOMALY → diagnose, classify tier, act per operating mode, record in experiment-context.md
       - COMPLETE → enter completion mode
    6. After any restart → enter intensive observation (60s interval, 5 cycles)
}
```
````

- [ ] **Step 2: Verify the replacement**

Read the file and confirm:
- The new loop uses `Bash tool: \`sleep\`` and `Bash tool: \`tail\`` (explicit tool calls)
- Startup grace period is present in step 3b
- Hang classification includes Tier 2/3 escalation path
- Step 5 NORMAL updates step time baseline
- Step 6 intensive observation is present

- [ ] **Step 3: Commit**

```bash
git add skills/watchdog/SKILL.md
git commit -m "feat(watchdog): rewrite monitoring loop with explicit bash sleep instructions"
```

---

### Task 3: Delete "Polling and Hang Detection" Section

**Files:**
- Modify: `skills/watchdog/SKILL.md` — delete the "Polling and Hang Detection" section

- [ ] **Step 1: Delete the section**

Remove the entire `## Polling and Hang Detection` section including its trailing blank line (from `## Polling and Hang Detection` through `...use a generous timeout (e.g., 15 minutes) before judging a hang.` plus the blank line after it), so that `## Restart Mechanism` follows with a single blank line separator. This covers:
- "Log as heartbeat" paragraph → now in Monitoring Loop step 2 parenthetical
- "Polling interval" list → now in Hard Gate + Monitoring Loop step 1
- "Must use Bash tool sleep" → now in Hard Gate
- "Hang detection" paragraph → now in Monitoring Loop step 3b
- "Step baseline" paragraph → now in Monitoring Loop step 5

- [ ] **Step 2: Verify no dangling references**

Search the file for any references to "Polling and Hang Detection" or content that assumed that section exists. Confirm none.

- [ ] **Step 3: Commit**

```bash
git add skills/watchdog/SKILL.md
git commit -m "refactor(watchdog): remove redundant polling section, content moved to hard gate and loop"
```

---

### Task 4: Sync plugin cache

**Files:**
- Source: `skills/watchdog/SKILL.md` (in git repo)
- Target: `~/.claude/plugins/spml/skills/watchdog/SKILL.md` (plugin install location)

- [ ] **Step 1: Copy updated file to plugin directory**

```bash
cp skills/watchdog/SKILL.md ~/.claude/plugins/spml/skills/watchdog/SKILL.md
```

- [ ] **Step 2: Sync to plugin cache if it exists**

```bash
if [ -d ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/watchdog ]; then
  cp skills/watchdog/SKILL.md ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/watchdog/SKILL.md
fi
```

- [ ] **Step 3: Verify the synced file contains the hard gate**

```bash
grep -c "HARD-GATE" ~/.claude/plugins/spml/skills/watchdog/SKILL.md
```

Expected: 2 (opening + closing tag, or more if existing hard gates)
