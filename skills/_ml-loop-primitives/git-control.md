# Primitive — Git Control

Supervisor-loop skills own all git write operations in a dedicated worktree. Researcher subagents have file-write and bash permissions but never commit, reset, or checkout.

## Rules

1. **Single writer.** Only the Supervisor performs `git commit`, `git reset`, `git checkout`, `git clean`. This guarantees a linear, auditable history.
2. **Work in a worktree.** Never run the loop on the main branch or in the main working directory. On startup:
   - Fresh: `git worktree add ../<skill>-<experiment_name> HEAD`
   - Resume: check `git worktree list`, reuse the existing worktree.
3. **.gitignore is a precondition.** Before the first round, verify `.gitignore` covers training artifacts (`outputs/`, `logs/`, `*.ckpt`, `wandb/`, etc.). If missing or incomplete, fix it before the loop starts. With a proper `.gitignore`, `git add -A` naturally skips artifacts.
4. **experiences.md survives rollback.** Before `git checkout -- . && git clean -fd`, copy `experiences.md` to `/tmp/`; restore it afterwards.

## Commit Pattern

```bash
cp experiences.md /tmp/experiences_backup.md
git add -A
git commit -m "<skill>: round {round} — <verdict_summary>"
```

## Rollback Pattern

```bash
cp experiences.md /tmp/experiences_backup.md
git checkout -- .
git clean -fd
cp /tmp/experiences_backup.md experiences.md
```

## Researcher Violation

If `git diff --name-only` after the Researcher finishes shows a locked file was modified, the round is a compliance failure. Roll back, record the violation in experiences.md Insight, continue to the next round.
