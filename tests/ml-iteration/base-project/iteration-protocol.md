---
mode: iteration
experiment: test-exp
max_rounds: 3
time_limit: 30s
train_command: STEPS=20 python train.py
eval_command: python evaluate.py
---

## Review criteria
metrics:
  - name: accuracy
    direction: ">="
    threshold: 0.95
performance:
  - first_step_time: "<= 1s"
observability:
  - "per-step loss output"
stability:
  - "no NaN"

## Modification boundary (soft)
- focused_files: [train.py]
- locked_files: [evaluate.py]
- Other: soft

## Initial hints
Increase STEPS to reach accuracy threshold.
