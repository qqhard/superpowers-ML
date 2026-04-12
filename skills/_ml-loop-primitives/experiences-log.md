# Primitive — Experiences Log

`experiences.md` is the shared memory of a Supervisor loop across rounds. Append-only. Two variants share the same header format and differ in per-round columns.

## Header

```markdown
# Experiences — <experiment_name>

- mode: iteration | autoresearch
- best_commit: <hash> (round {N})
- best_state:
    {metric_or_criteria_snapshot}
- rounds: {completed_round_count}
- status: not_started | running | completed | target_reached | stopped_by_user

---
```

The Supervisor updates `best_commit`, `best_state`, `rounds`, and `status` after each round.

## Row (autoresearch variant)

| Round | Strategy | {metric} | Verdict | Insight |
|-------|----------|----------|---------|---------|
| 1 | cosine lr + label smoothing | 0.813 | committed | label smoothing improved over warmup |
| 2 | mixup aug | 0.799 | rolled_back | mixup hurt early-epoch accuracy |

## Row (iteration variant)

| Round | Strategy | Metrics | Speed | Observability | Stability | User Hint | Verdict | Insight |
|-------|----------|---------|-------|---------------|-----------|-----------|---------|---------|
| 1 | ... | ... | ... | ... | ... | "next focus: log format" | committed | ... |

Each column corresponds to a `review_criteria` dimension (plus a `User Hint` column for human-on-the-loop input). Dimensions absent from the design doc's review_criteria get omitted columns.

## Discipline

- **Append-only within a round.** Once a verdict is recorded, the row is not rewritten. Supervisor-override of a prior verdict creates a new row describing the override.
- **Researcher writes the Strategy column only.** Everything else (metrics, verdict, insight) is written by the Supervisor.
- **Insight explains cause, not outcome.** For rolled-back rounds, the Insight must help the next round's Researcher avoid the same failure.
