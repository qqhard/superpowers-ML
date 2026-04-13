# Primitive — Scheduling Safety Net

Four-layer CronCreate mechanism that prevents Supervisor loops from stalling silently when background tasks hang or the REPL goes idle.

## Layers

1. **Task-completion notification.** Dispatch long tasks with `run_in_background: true`. When the task finishes, the Supervisor is automatically notified and continues. Applies to Researcher dispatches, training runs, eval runs.

2. **Check-in reminder.** After dispatching ANY background task, immediately create a one-shot `CronCreate` to wake the Supervisor in ~120 seconds (adjust based on expected duration). When it fires:
   - If the task completed, continue.
   - If still running, create another reminder.
   - Delete the reminder when the task completes normally.

3. **Per-round timeout timer.** Before a round starts, create a one-shot `CronCreate` with `time_limit * 2`. If the round hasn't completed when it fires, mark the round failed and move on. Delete when the round completes normally.

4. **Session heartbeat.** A recurring `CronCreate` (every 30 minutes) as ultimate safety net. Fires only when REPL is idle. Self-audits: "do I have a task list? is a background task running? am I stalled?" Delete when the loop terminates.

## Hard Rule

**Never say "I'll wait" without creating a timer.** If you dispatch a background task and intend to check back later, you MUST create a `CronCreate` one-shot immediately. "I'll check in 2 minutes" without a timer is a bug — the REPL goes idle and nothing wakes you up until the task completes or the 30-minute heartbeat fires.

## Heartbeat Prompt Template

```
Supervisor heartbeat — self-audit:
- Current round has task list? If no, rebuild now.
- Writing code yourself? Stop — dispatch subagent.
- Background task running or round just finished? If neither, you're stalled — resume.
- Waiting for user confirmation? Don't — advance autonomously; user is on the loop, not in it.
Never skip steps for "simplicity". Then continue.
```

## Cleanup

On loop termination (target reached / max_rounds / user stop), delete all Supervisor-owned CronCreate jobs (by saved IDs) before the final report.
