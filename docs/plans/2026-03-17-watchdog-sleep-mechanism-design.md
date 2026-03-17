# Watchdog Sleep Mechanism Design

**Date:** 2026-03-17
**Status:** Draft
**Scope:** `skills/watchdog/SKILL.md` only

## Problem

The Watchdog skill describes an adaptive monitoring loop with sleep-based intervals, but in practice Claude does not execute `bash sleep` between checks. Instead it either:

1. Outputs "next check in 2 minutes" and stops — no timer, no wake-up, nothing happens
2. When prompted, writes a Python monitoring script — violating the skill's intent and adding unnecessary complexity

The root cause: the sleep instruction is buried in a subsection ("Polling and Hang Detection", line 136) as a single sentence. The main monitoring loop uses abstract pseudocode (`Sleep (interval)`) that Claude interprets figuratively, not as a literal Bash tool call.

## Design

Three changes to `skills/watchdog/SKILL.md`:

### 1. New Hard Gate: Sleep Loop Mechanism

Add a new `<HARD-GATE>` block before the Operating Modes section, ensuring all three modes (Monitor, Guardian, Autonomous) are bound by it.

```markdown
<HARD-GATE>
## Monitoring Loop Mechanism

You MUST use the Bash tool to execute `sleep <seconds>` as the ONLY way to wait between checks.

After each sleep returns, immediately proceed to the next check — do NOT wait for user input.

**Prohibited:**
- Outputting "I'll check in N minutes" and then stopping
- Writing a monitoring/watchdog script (Python, shell, or any other)
- Using /loop or any external scheduling mechanism
- Asking the user to remind you to check

**Required execution pattern:**
1. Bash tool: `sleep 120`  (or appropriate interval)
2. Bash tool: `tail -20 <log_file>`  (read latest log lines)
3. Analyze output, report status
4. If anomaly → diagnose and act per operating mode
5. Go to step 1
</HARD-GATE>
```

### 2. Rewrite Monitoring Loop

Replace the current abstract pseudocode with implementation-level description:

```markdown
## Monitoring Loop

loop {
    1. Bash tool: `sleep <interval_seconds>`
       - Normal: 120-300s (2-5 min)
       - Post-restart / post-anomaly: 60s for 5 cycles, then back to normal
    2. Bash tool: `tail -20 <log_file>`
    3. Check for new lines since last check:
       a. New lines → parse metrics, go to step 4
       b. No new lines → Bash tool: `ps aux | grep <training_script>`
          - Process dead → read exit code → classify problem tier → act
          - Process alive → compare silence duration vs step baseline
            - Within baseline → continue (go to step 1)
            - Exceeds 10x baseline → kill process → classify as Tier 1
    4. Analyze metrics:
       a. Sanity: NaN, Inf, negative loss, zero gradient
       b. Baseline comparison vs VP ranges in experiment-context.md
       c. Trend: loss decreasing? grad_norm stable?
    5. Classify:
       - NORMAL → output one-line progress, go to step 1
       - ANOMALY → diagnose, classify tier, act per operating mode, record in experiment-context.md
       - COMPLETE → enter completion mode
}
```

### 3. Delete "Polling and Hang Detection" Section

The current standalone section (lines 129-142) is redundant — its content is now covered by:
- Sleep intervals → Hard Gate + Monitoring Loop step 1
- Hang detection → Monitoring Loop step 3b
- Step baseline → retained in Monitoring Loop context

The one-liner about `sleep` on line 136 ("Must use Bash tool `sleep` to implement intervals") is superseded by the Hard Gate.

## Non-Changes

The following are explicitly **not** in scope:
- `skills/training-handoff/SKILL.md` — no interface change; watchdog-prompt.md format unchanged
- `skills/training-resume/SKILL.md` — consumes recovery/completion prompts, unaffected
- `skills/experiment-planning/SKILL.md` — upstream of Watchdog, unaffected
- Operating modes (Monitor/Guardian/Autonomous) — unchanged
- Problem classification (Tier 1/2/3) — unchanged
- Diagnosis, Completion, and Common Anomaly Patterns sections — unchanged
