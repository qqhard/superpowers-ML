# Primitive — Researcher Subagent Dispatch

Pattern used by `ml-iteration` and `autoresearch` to hand off code modification to a subagent, then resume the Supervisor loop when the subagent returns.

## Principles

- **Fresh subagent per round.** No shared agent memory between rounds. Experience transfer happens through files (`experiences.md`, git history).
- **Supervisor injects lightweight context, not file contents.** The Researcher reads files itself — this keeps the Supervisor's context clean.
- **Background dispatch with timer pairing.** Every Researcher dispatch creates two timers (check-in + per-round timeout). Both are deleted when the Researcher completes normally.

## Prompt Skeleton

```
You are an ML researcher. Your task is to {task_description}.
This is Round {round} of {max_rounds}.

## Your role
Design a strategy and modify code. Training and evaluation are run by Supervisor
after you finish; you may smoke-test your code but do not run full training yourself.

## Constraints
- Do NOT modify any evaluation logic. Eval is a pre-defined script Supervisor owns.
- Soft boundary (ml-iteration) or Fixed/Variable (autoresearch) — {boundary_rules}.

## Recent experiences
{last_M_rounds_table_snippet}

## Task
1. Read the relevant files to understand current code.
2. Design a strategy for this round.
3. Append a row to {experiences_path} with your strategy (leave Result/Verdict blank).
4. Modify the code accordingly.
5. Report "Code ready" as your final message.
```

## Dispatch Mechanics

- Use `Agent` tool with `run_in_background: true`.
- Immediately after dispatching, create two `CronCreate` one-shots:
  - Check-in reminder (~120s) — prompts self-check whether Researcher completed.
  - Per-round timeout (`time_limit * 2`) — fails the round if Researcher hangs past budget.
- Save both job IDs. Delete both on normal completion.

## Timeout Handling

If per-round timeout fires before Researcher returns, treat the round as failed. Record diagnosis in `experiences.md` Insight, roll back, continue to next round.
