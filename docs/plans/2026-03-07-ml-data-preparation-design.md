# ml-data-preparation Design Document

**Date:** 2026-03-07

**Positioning:** Independent skill for dataset processing TDD workflow. Validates data processing logic correctness on small-scale data before running full-scale. Independent of the training Validation Pyramid.

---

## 1. Trigger Conditions

**When to use:**
- Constructing or transforming a dataset for training use
- When `ml-brainstorming` identifies dataset processing work

**When NOT to use:**
- Training-time data loading performance issues (belongs to Validation Pyramid L0)
- Dataset already exists and can be used directly without processing

---

## 2. Core Flow

```
Confirm target framework + format
    |
    v
Small-scale sample (100-1000 rows)
    |
    +-- Step 1: Write tests (TDD first)
    |     - Target framework can load correctly (format compliance)
    |     - Read efficiency is acceptable (benchmark)
    |     - Field types, shapes, encoding are correct
    |
    +-- Step 2: Implement data processing code
    |     - Make Step 1 tests pass
    |     - Follow data operation safety principles
    |
    +-- Step 3: Small-scale data analysis
    |     - Distribution statistics, spot check, dedup
    |     - Human review a few samples
    |     - Confirm processing logic is correct
    |
    All tests pass + distribution meets expectations
    |
    v
Run full-scale (with progress viewing instructions for user)
```

**Key principle:** Do NOT run full-scale until small-scale validation passes.

---

## 3. Step 1: Write Tests (Format Compliance + Read Efficiency)

| Check | Description |
|-------|-------------|
| Target framework loading | Actually load with target DataLoader, no errors |
| Field completeness | All expected fields present, no extra fields |
| Type correctness | dtype, shape match model input requirements |
| Encoding correctness | Text encoding, categorical mapping, special tokens correct |
| Read efficiency | Benchmark single batch load time, confirm not a bottleneck |

---

## 4. Step 2: Implement Data Processing Code

Make Step 1 tests pass, following these safety principles:

### Data Operation Safety

Data operations (add, delete, modify) are inherently dangerous. Small-scale validation exists to:
- Avoid large resource waste from incorrect logic
- Avoid unintended destructive operations (e.g., accidental deletion)

**Safety rules:**
- **Copy-first:** Execute add/delete/modify on a small-scale copy first, confirm results correct, then operate on original data
- **Destructive operations require confirmation:** Delete, overwrite operations must not auto-execute — require explicit user confirmation
- **Prefer generating new files** over in-place modification

### Progress Display Requirements

Data processing code MUST include progress display:
- Use `tqdm` or equivalent progress bars
- Log key milestones (e.g., percentage complete, ETA)
- Write checkpoints for resumability on long-running jobs

---

## 5. Step 3: Small-Scale Data Analysis

| Check | Description |
|-------|-------------|
| Label distribution | Compare with original data source, confirm processing didn't introduce bias |
| Feature distribution | Key features' min/max/mean/std, check for anomalies |
| Missing rate | Per-field missing ratio within expected range |
| Sample spot check | Human review 5-10 samples, confirm end-to-end correctness |
| Dedup check | Confirm no unexpected duplicate samples |

---

## 6. Running Full-Scale

After small-scale validation passes, before launching full-scale processing:

1. Agent provides the user with a **progress viewing guide**:
   - Where to see output / logs
   - Estimated run time
   - How to check progress (e.g., `tail -f logs/processing.log`, tqdm output)
   - How to resume if interrupted

2. Agent does NOT poll or monitor the long-running task (avoids token waste)

---

## 7. What This Skill Does NOT Do

| Out of scope | Reason |
|--------------|--------|
| Full-scale post-verification | Time-consuming; small-scale validation is sufficient to confirm logic correctness |
| Agent polling long-running tasks | Token waste; user monitors via progress display |
| Training-time data loading issues | Belongs to Validation Pyramid L0 Engineering Efficiency |
| Data collection / crawling | Separate concern, not dataset processing |
