# Steady-State Efficiency: MFU / TCA / Sample Speed

**IMPORTANT: All measurements MUST be taken during steady-state training.** Measurements during warmup (CUDA lazy init, JIT compilation, data prefetch filling) are not representative.

## Measurement Protocol

1. Run training normally
2. Skip the first 5-10 steps (warmup)
3. Confirm data loading is in steady state (prefetch buffers full)
4. Measure over the next N steps (N >= 10)
5. Report: **sample speed**, **TCA**, **MFU**

## MFU (Model FLOPs Utilization)

MFU = Actual Model FLOPs / Theoretical Peak FLOPs

**Using toolkit (when available):**
```python
from toolkit.profiling.mfu_calculator import calculate_mfu

result = calculate_mfu(model, input_shape=(batch_size, seq_len), step_time_ms=measured_step_time)
print(f"MFU: {result['mfu']:.2%}, TCA: {result['tca']:.2%}")
assert result['mfu'] >= target_mfu, f"MFU {result['mfu']:.2%} below target {target_mfu:.2%}"
```

**Manual calculation:**
```python
import torch
import time

# 1. Count model FLOPs (approximate for Transformer)
def estimate_transformer_flops(num_params, seq_len, batch_size):
    """Approximate: 6 * num_params * seq_len * batch_size per step (fwd + bwd)"""
    return 6 * num_params * seq_len * batch_size

# 2. Measure step time — skip warmup, measure steady-state
warmup_steps = 10
measure_steps = 20

for _ in range(warmup_steps):
    output = model(mock_input)
    loss = criterion(output, mock_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(measure_steps):
    output = model(mock_input)
    loss = criterion(output, mock_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
step_time = (time.perf_counter() - start) / measure_steps

# 3. Sample speed
samples_per_sec = batch_size / step_time
tokens_per_sec = batch_size * seq_len / step_time
print(f"Sample speed: {samples_per_sec:.1f} samples/sec, {tokens_per_sec:.0f} tokens/sec")

# 4. Get GPU peak FLOPS
# A100: 312 TFLOPS (bf16), H100: 989 TFLOPS (bf16)
gpu_peak_tflops = 312  # Adjust for your GPU

# 5. Calculate MFU
model_flops = estimate_transformer_flops(num_params, seq_len, batch_size)
mfu = model_flops / (gpu_peak_tflops * 1e12 * step_time)
print(f"MFU: {mfu:.2%}")
```

## Memory Analysis

**Using toolkit (when available):**
```python
from toolkit.profiling.memory_profiler import analyze_memory
result = analyze_memory(model, mock_input)
print(result)
```

**Manual check:**
```python
torch.cuda.reset_peak_memory_stats()

output = model(mock_input)
loss = criterion(output, mock_target)
loss.backward()

allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
peak = torch.cuda.max_memory_allocated() / 1e9

print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved:  {reserved:.2f} GB")
print(f"Peak:      {peak:.2f} GB")
print(f"Fragmentation: {(reserved - allocated) / reserved:.1%}")

# Flag if fragmentation > 20%
assert (reserved - allocated) / reserved < 0.2, "High memory fragmentation detected"
```

## Per-Layer Profiling (on failure)

**Using toolkit (when available):**
```python
from toolkit.profiling.layer_profiler import profile_layers
result = profile_layers(model, mock_input)
for layer in result['layers']:
    print(f"{layer['name']}: {layer['time_ms']:.2f}ms ({layer['percentage']:.1f}%)")
```

**Manual:**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    output = model(mock_input)
    loss = criterion(output, mock_target)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```
