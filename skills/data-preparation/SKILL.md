---
name: data-preparation
description: Use when constructing or transforming datasets for training - guides TDD-first workflow to validate data processing logic on small-scale data before running full-scale
---

# Data Preparation: TDD-First Dataset Processing

## Overview

Before building a full dataset, validate your data processing logic on a small scale. Write tests first, implement processing code, analyze the output, then scale up. This prevents wasting hours/days on incorrect processing logic and avoids dangerous data operations (accidental deletion, corruption).

**Core principle:** Do NOT run full-scale until small-scale validation passes.

<HARD-GATE>
Do NOT run full-scale data processing until:
1. Tests are written and passing on small-scale data
2. Small-scale data analysis confirms correctness
3. User has explicitly approved scaling up
</HARD-GATE>

## When to Use

- Constructing or transforming a dataset for training use
- Converting data to a specific format for a target framework/DataLoader
- Any data pipeline that processes raw data into training-ready format

## When NOT to Use

- Dataset already exists and needs no processing
- Training-time data loading performance issues (use L0 Engineering Efficiency)
- Data collection / crawling (separate concern)

## Checklist

You MUST complete these in order:

1. **Confirm target format** — what framework reads this data, what format does it need
2. **Prepare small-scale sample** — extract 100-1000 rows from source data
3. **Write tests (TDD first)** — format compliance + read efficiency + field correctness
4. **Implement data processing code** — make tests pass, follow safety principles
5. **Run small-scale data analysis** — distribution, spot check, dedup
6. **User approves results** — present analysis, get explicit approval
7. **Run full-scale** — with progress viewing instructions for user
8. **Commit**

## Step 1: Confirm Target Format

Ask the user (one question at a time):
- What framework/DataLoader will consume this data? (PyTorch DataLoader, HuggingFace datasets, TFRecord, etc.)
- What format? (parquet, jsonl, tfrecord, arrow, csv, binary, etc.)
- What fields/columns are needed? (features, labels, metadata)
- What dtypes/shapes? (match model input requirements)

## Step 2: Prepare Small-Scale Sample

- Extract 100-1000 representative rows from source data
- **Work on a COPY, never the original**
- Ensure sample covers edge cases: missing values, special characters, boundary values

## Step 3: Write Tests (TDD First)

Write these tests BEFORE writing any data processing code:

### Format Compliance Tests

```python
def test_target_framework_can_load():
    """Target DataLoader loads the processed data without errors."""
    dataset = load_with_target_framework("output/small_scale/")
    assert len(dataset) > 0

def test_all_expected_fields_present():
    """All required fields exist, no unexpected extra fields."""
    sample = dataset[0]
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}  # adapt to actual fields

def test_field_dtypes_correct():
    """Each field has the correct dtype and shape."""
    sample = dataset[0]
    assert sample["input_ids"].dtype == torch.long
    assert sample["input_ids"].shape == (max_seq_len,)

def test_encoding_correct():
    """Text encoding, categorical mapping, special tokens are correct."""
    sample = dataset[0]
    # Verify specific encoding rules for your data
```

### Read Efficiency Tests

```python
def test_batch_load_time_acceptable():
    """Single batch loads within acceptable time."""
    import time
    dataloader = create_dataloader("output/small_scale/", batch_size=32)
    start = time.perf_counter()
    batch = next(iter(dataloader))
    elapsed = time.perf_counter() - start
    print(f"Single batch load time: {elapsed:.4f}s")
    assert elapsed < 1.0  # adjust threshold per use case
```

**Run tests to verify they FAIL** (processing code doesn't exist yet).

## Step 4: Implement Data Processing Code

Write the data processing code to make Step 3 tests pass.

### Data Operation Safety Principles

Data operations (add, delete, modify) are inherently dangerous. Follow these rules:

| Rule | Why |
|------|-----|
| **Copy-first** | Execute on a small-scale copy first, confirm correct, then operate on original |
| **Destructive ops need confirmation** | Delete/overwrite must not auto-execute — require explicit user confirmation |
| **Generate new files, don't modify in-place** | Prefer writing to a new output directory over overwriting source files |

### Progress Display Requirements

Data processing code MUST include:
- `tqdm` or equivalent progress bars
- Key milestone logging (percentage complete, ETA)
- Checkpoint support for resumability on long-running jobs

```python
from tqdm import tqdm

def process_dataset(input_path, output_path):
    raw_data = load_raw(input_path)
    processed = []
    for item in tqdm(raw_data, desc="Processing"):
        processed.append(transform(item))
        # Checkpoint every N items for resumability
    save(processed, output_path)
```

**Run tests to verify they PASS.**

## Step 5: Small-Scale Data Analysis

After tests pass, analyze the processed small-scale data:

| Check | How |
|-------|-----|
| Label distribution | Compare with original source, confirm no introduced bias |
| Feature distribution | Key features' min/max/mean/std, check for anomalies |
| Missing rate | Per-field missing ratio within expected range |
| Sample spot check | Print 5-10 samples, human review end-to-end correctness |
| Dedup check | Confirm no unexpected duplicate samples |

```python
def analyze_processed_data(dataset):
    """Print analysis for human review."""
    print(f"Total samples: {len(dataset)}")
    print(f"\n--- Label Distribution ---")
    # label counts / percentages
    print(f"\n--- Feature Statistics ---")
    # min, max, mean, std for key features
    print(f"\n--- Missing Rate ---")
    # per-field missing count / percentage
    print(f"\n--- Sample Spot Check (first 5) ---")
    for i in range(min(5, len(dataset))):
        print(f"\nSample {i}: {dataset[i]}")
    print(f"\n--- Duplicate Check ---")
    # check for exact duplicates
```

**Present results to user. Get explicit approval before proceeding to full-scale.**

## Step 6: Run Full-Scale

After user approves small-scale results:

1. **Provide progress viewing guide to user:**

```
Running full-scale data processing.

Estimated time: [X hours/minutes]
Monitor progress: tail -f logs/processing.log
Progress bar: visible in terminal via tqdm
Output location: [path]
Resume if interrupted: re-run the same command, it will skip completed chunks
```

2. Launch the full-scale processing
3. **Agent does NOT poll or monitor** — user watches progress themselves

## Common Failure Patterns

| Symptom | Likely Cause |
|---------|-------------|
| DataLoader throws error | Format mismatch: check dtype, field names, file structure |
| Fields missing | Transform logic drops fields, or source data has unexpected schema |
| Wrong dtype | Encoding step produces wrong type, or missing cast |
| Distribution skewed vs source | Filter/sampling logic introduces bias |
| Duplicates appear | Join/merge logic creates cartesian product |
| Load time too slow | Wrong file format for access pattern, missing indexing, no chunking |
