# ML Iteration Skill + Watchdog Refactor — Design

## Problem

Two current skills have significant conceptual overlap:

- `watchdog` — long-running supervisor for a single training run. Monitors logs, classifies anomalies into three tiers, can auto-restart on env failures, auto-fix simple parameters, or spawn a sub-agent for complex code fixes.
- `autoresearch` — long-running Supervisor for N rounds of metric-driven search. Each round dispatches a fresh Researcher subagent, runs training + eval, commits on improvement or rolls back on regression.

Both dispatch training, run eval, manage git state, maintain a history file, and handle scheduling/timeout. Watchdog's Tier-2 (parameter fix + restart) and Tier-3 (code fix + restart) are effectively degenerate single-round search loops implemented inside what is supposed to be a stability monitor.

Separately, ML work rarely succeeds in one training run. Even outside of metric search, a handoff flow almost always needs a "run → review → tweak → rerun" loop — to improve speed, fix logging, sharpen evaluation, or patch unfinished pieces. Today the only post-handoff option besides watchdog is autoresearch, which requires committing to rigid Fixed/Variable file partitions and a single metric target. There is no skill that supports compound-criteria iteration with a human reviewing each round.

## Design Goals

1. Give `watchdog` a single clear responsibility: keep a single training run healthy. Strip anything that crosses the "let this training finish" boundary.
2. Introduce `ml-iteration` as the post-handoff N-round loop for non-search cases. Human on the loop; Supervisor acts as Reviewer against compound criteria established at design time.
3. Keep `autoresearch` unchanged. Its rigid protocol is the right shape for metric search and has no reason to move.
4. Route handoff between the two non-search paths (`watchdog` or `ml-iteration`) by asking at handoff time. Autoresearch keeps its separate entry (`autoresearch-create`).
5. Share only what should be shared. Extract common patterns (Researcher dispatch, scheduling, git control, experiences log, eval lock) into reference documents the skills cite, not a new skill.

## Architecture Overview

```
ml-brainstorming → experiment-planning → ml-subagent-dev → training-handoff
                                                                │
                                                   ┌────────────┴────────────┐
                                                   ▼                         ▼
                                              watchdog                 ml-iteration
                                      (single-run supervision)    (N-round human-on-loop)

autoresearch-create → ml-brainstorming (autoresearch mode)
                   → experiment-planning
                   → ml-subagent-dev
                   → autoresearch-handoff
                   → autoresearch                (metric search; unchanged)
```

### Responsibility Split

| Skill | Keeps one training run healthy | Drives N-round iteration | Decides "what to change next" |
|---|---|---|---|
| `watchdog` (narrowed) | ✅ ckpt-restart on env failure, async eval, baseline alerts | ❌ | ❌ |
| `ml-iteration` (new) | ❌ bare `Bash` — env crash fails the round | ✅ until criteria met / max_rounds / user stop | Supervisor-as-Reviewer + optional human input |
| `autoresearch` (unchanged) | ❌ bare `Bash` | ✅ until target / max_rounds / user stop | Researcher picks strategy; Supervisor judges via Pareto rule |

### Human on the Loop, Not In

`ml-iteration` is not "human-in-the-loop with approval at every round." The Supervisor runs autonomously and is itself the Reviewer. The human watches, and can interject any time — supplement criteria, override a verdict, aim the next round — but default execution does not block on the human. This mirrors `autoresearch`'s principle, but the review is compound and LLM-judged instead of single-metric Pareto.

## Compound Review Criteria

Collected during `ml-brainstorming` and written into the experiment design doc:

```yaml
review_criteria:
  metrics:            # eval-output numeric thresholds
    - name: accuracy
      direction: ">="
      threshold: 0.85
  performance:        # speed / efficiency expectations visible while running
    - first_step_time: "<= 30s"
    - mfu: ">= 0.30"
  observability:      # log / progress-bar / debugging expectations
    - "per-step loss / grad_norm / step_time"
    - "tqdm progress bar per phase"
  stability:          # process-health expectations
    - "no NaN / Inf"
    - "no torch autograd warnings"
  custom:             # anything else raised in brainstorming
    - "checkpoint is HF AutoModel compatible"
```

Any field may be empty; brainstorming records what the user actually cared about. The structure is deliberately not a strict enum — ML "good-enough-to-hand-off" is context-dependent, and the Supervisor needs to be free to judge holistically.

`review_criteria` is written to the design doc even when the user will run `watchdog` (not iterate). It serves as the compass for `verification` to judge overall experiment success later.

## Supervisor-as-Reviewer (ml-iteration)

After each round's training + eval, the Supervisor produces a review:

1. Walk each criterion: pass / fail / improved-vs-last-commit / regressed.
2. Emit an overall verdict:
   - **all criteria pass** → `commit` + terminate (done)
   - **no criterion regressed and at least one improved** → `commit`
   - **any criterion clearly regressed** → `rollback`
   - **ambiguous** → `accept_with_note` (commit, flagged in experiences)
3. Present the review to the user; continue executing without waiting.
4. The user may interject asynchronously:
   - `stop` → finish current step, terminate
   - protocol edit (e.g. "also watch expert utilization") → Supervisor edits `iteration-protocol.md` in place, continues
   - next-round hint ("next round focus on log format") → append to current round's User Hint field, injected into next Researcher's prompt
   - verdict override ("roll back that last commit") → Supervisor re-executes git operations accordingly

Judgment is LLM-native, not rule-based. The rationale: compound criteria are inherently multi-dimensional and context-dependent, and Pareto rules produce false rollbacks for changes that trade one dimension for another in ways the human would have approved. Keeping the verdict LLM-produced but surfaced to the human preserves both speed (default autonomous) and correctness (human can override cheaply).

## Round Flow

```
for round N in 1..max_rounds:

  0. Create S1–S6 task list for this round.
  1. Assemble Researcher prompt:
       - review_criteria (full)
       - current best commit's state across each dimension
       - last M rounds of experiences (including rolled-back rounds as negative examples)
       - latest user hint (if any)
       - soft boundary: focused_files, locked_files, other=soft
  2. Dispatch Researcher (run_in_background). Researcher modifies code, reports strategy.
  3. Compliance check: `git diff --name-only` vs locked_files. Violation → rollback + flag.
  4. Run train_command (bare Bash, within time_limit). Env crash fails the round.
  5. Run eval_command (locked — Researcher cannot have modified or replaced it).
  6. Supervisor generates compound review and verdict (see above).
  7. Execute git operation per verdict; preserve experiences.md across rollback.
  8. Absorb any user input since last round into the next Researcher's prompt.
  9. Termination check: all criteria met / max_rounds / user stop.
```

## Protocol File

Generated by `training-handoff` into the experiment directory:

**File:** `iteration-protocol.md`

```yaml
---
mode: iteration
experiment: <name>
max_rounds: 10
time_limit: 10min          # per-round training budget; observation over completion
train_command: <from VP L1>
eval_command: <locked, from VP L1>
---

## Review criteria
<copied verbatim from design doc>

## Modification boundary (soft)
- Focused_files: [...]           # from brainstorming
- Locked_files: [<eval path>, <core data loader>]   # hard constraint
- Other: soft — Researcher may modify, recorded in experiences

## Initial hints
<optional, from brainstorming wrap-up>
```

Deliberately **not** the same schema as `autoresearch-protocol.md`. Their semantics diverge enough that merging schemas would produce optional-heavy, interpretation-heavy fields. What the two share is the **pattern**: protocol is single source of truth, Researcher never reads it, Supervisor injects slices into prompts.

## Handoff Routing

`training-handoff` extends to ask at decision time:

> "This experiment has been validated. Two ways to run from here:
>
> 1. **watchdog** — run training once, auto-restart on environment failures, async eval on new checkpoints.
> 2. **ml-iteration** — run N short rounds (≈ 10 min each), Supervisor reviews against your review_criteria and commits improvements; you can interject any time.
>
> Which?"

The choice is recorded in the experiment directory. Default suggestion is `ml-iteration` when `review_criteria` has at least one non-empty dimension; `watchdog` when the user explicitly declared a single-run intent in brainstorming.

`autoresearch-create` remains the only entry into autoresearch. No handoff branch to it — its setup requires a stricter protocol built earlier.

## Naming

| Role | Name |
|---|---|
| New N-round iteration skill | `ml-iteration` |
| Protocol file | `iteration-protocol.md` |
| Experience log (shared format with autoresearch) | `experiences.md` |
| Handoff skill | `training-handoff` (extended, not split) |

`ml-iteration` matches the `ml-brainstorming` / `ml-subagent-dev` convention. `supervisor` is an implementation concept, not a user-facing one.

## Watchdog Narrowing

Behavior to **remove** from `watchdog`:

- Tier 2: automatic parameter fix + restart
- Tier 3 / Autonomous mode: sub-agent spawn to fix code + restart
- The three-mode system (Monitor / Guardian / Autonomous) — collapse to a single mode

Behavior to **keep**:

- Tier 1: environment-failure detection (OOM, NCCL, disk, SIGKILL, hang) → ckpt-restart
- Async evaluation on new checkpoints → eval subagent → append summary to experiment-context.md
- Baseline-deviation alerts (MFU drop, loss anomaly) → report only, no auto-fix

Anything that crosses "let this training finish" — changing parameters, changing code — is now out of scope for `watchdog` and in scope for `ml-iteration`. If a user wants that behavior, they pick `ml-iteration` at handoff time.

## Shared Primitives

Extracted as **reference documents**, not a new skill:

```
skills/_ml-loop-primitives/
├── researcher-dispatch.md      # Agent tool dispatch, prompt assembly, timer cleanup
├── scheduling-safety-net.md    # Four CronCreate layers (task-completion / check-in / per-round-timeout / heartbeat)
├── git-control.md              # Supervisor-only git writes, worktree discipline, experiences.md backup, .gitignore prerequisites
├── experiences-log.md          # experiences.md columns (machine / human variants), append-only discipline
└── eval-lock.md                # eval script lives in locked files; Researcher cannot create alternative eval logic
```

The leading underscore marks the directory as skill-author reference material, not a user-invokable skill.

| Primitive | watchdog | ml-iteration | autoresearch |
|---|---|---|---|
| `scheduling-safety-net.md` | ✅ | ✅ | ✅ |
| `git-control.md` | n/a (no multi-round git writes) | ✅ | ✅ |
| `researcher-dispatch.md` | n/a (no Researcher) | ✅ | ✅ |
| `experiences-log.md` | n/a | ✅ | ✅ |
| `eval-lock.md` | n/a (no eval decisions) | ✅ | ✅ |

Primitives are documentation; each skill still contains the procedure that uses them. This keeps each skill self-contained and readable, while removing duplicate language across the three SKILL.md files.

## Anomaly Handling (ml-iteration)

| Anomaly | Handling |
|---|---|
| Researcher timeout / crash | Round fails, rollback, insight records cause; no automatic retry |
| Training env crash (OOM, NCCL) | Round fails; if env instability is persistent, the user can stop ml-iteration and re-handoff into watchdog mode |
| Training exceeds time_limit | Script should exit cleanly before; Bash timeout (+ buffer) is backstop, treated as env crash |
| eval_command fails | Supervisor fixes environment / paths only; **never** modifies eval logic; unfixable → pause + notify user |
| N consecutive rollbacks | Plateau warning surfaced; continues (not a termination condition) |
| Locked_files violation | rollback + experiences flag; does not count as a valid round |

## Migration Path

Ordered by risk and dependency:

1. **Create `skills/_ml-loop-primitives/` reference documents.** No functional change.
2. **Narrow `watchdog`.** Remove Tier 2 / Tier 3 / three-mode system. Retain Tier 1, async eval, baseline alerts. This is a behavior-contract change — running Watchdog tasks that relied on auto-parameter-fix will no longer get that behavior. Release notes must call out the change and point users at `ml-iteration`.
3. **Extend `ml-brainstorming`** to collect `review_criteria`. Backward-compatible: if an older design doc lacks the field, `training-handoff` prompts the user to fill it in at handoff time.
4. **Write `ml-iteration` skill** plus its agents/prompts. Additive.
5. **Extend `training-handoff`** to ask watchdog vs ml-iteration and generate the right protocol. Backward-compatible: the "always watchdog" default preserves current behavior when the user does not choose.
6. **README update.** New top-level capability entry for `ml-iteration`; reshape the Auto Research section's comparison table.

### Out of Scope for This Change

- `autoresearch`, `autoresearch-create`, `autoresearch-handoff` — unchanged. They are working, and their protocol rigidity is the right design for metric search.
- `verification` skill — may later consume `review_criteria` for a stronger final verdict, but that is a separate improvement.
- Cross-conversion tools between `iteration-protocol.md` and `autoresearch-protocol.md`. YAGNI.

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Handoff destinations | 2 (watchdog / ml-iteration) + autoresearch via its own entry | Three destinations would require brainstorming to decide early; autoresearch already has a dedicated entry, so handoff only branches between two |
| Watchdog + Supervisor composition | Peers, no nesting | Nesting re-entangles "keep running" and "iterate" responsibilities; the decoupled architecture costs the occasional lost round to env crashes but keeps the model simple |
| Commit/rollback policy | Supervisor-judged compound review + human override | Pareto rules produce false rollbacks on trade-offs; full human-in-loop defeats the autonomy intent; LLM judgment with human veto is the middle ground |
| Shared layer | Reference docs, not a skill | Skills are user-facing; primitive reference is author-facing. A shared skill would invite misinvocation |
| Protocol schema unification | Keep separate schemas | Semantics diverge enough that merging creates optional-heavy fields; pattern is shared, fields are not |
| Watchdog modes | Collapse three modes to one | Monitor/Guardian/Autonomous existed to express "how much the Watchdog should do"; after narrowing, there is only one answer |
| autoresearch changes | None | Working, orthogonal, risk not worth the marginal symmetry gain |
| Skill name | `ml-iteration` | Consistent with `ml-brainstorming` / `ml-subagent-dev`; `supervisor` is an implementation detail |
