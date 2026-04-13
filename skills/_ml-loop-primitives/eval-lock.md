# Primitive — Eval Lock

The evaluation script is a pre-defined, deterministic program. Neither the Researcher nor the Supervisor may modify its logic during the loop.

## Why

If the Researcher could write or change eval code, it could (intentionally or not) produce favorable metrics that mislead the loop. Locking eval is the single largest protection against agent self-deception in autonomous ML iteration.

## Mechanics

- Eval script path is fixed at protocol-generation time (during `training-handoff` or `autoresearch-handoff`) and recorded in the protocol file.
- The script lives in **locked files** (iteration) or **Fixed.files** (autoresearch). Any modification is a compliance violation.
- **Supervisor never substitutes training-log metrics for eval.** The eval command is the only source of truth. If it fails, fix the environment, paths, or missing deps — never the eval logic.
- If the Researcher creates any new eval-like script (even in new files that are not nominally locked), the Supervisor ignores those and uses the original `eval_command` only.

## Enforcement Hook

In the Supervisor's compliance check, beyond the `git diff --name-only` check for locked files, also grep for common eval-function names (`evaluate`, `compute_metrics`, `accuracy`, `score`) in any newly created files. Flag matches for user review.

## When the Eval Needs to Change

If the eval script itself is buggy or must evolve, the loop must be stopped. Changes to eval logic belong in a new experiment or a new handoff cycle, not in the middle of a running loop. This is not a limitation — it is the guarantee that every round's metric is comparable.
