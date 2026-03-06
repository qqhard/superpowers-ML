# Backend Checks

## FlashAttention

```python
# Check if FlashAttention is being used
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Method 1: Check via torch profiler
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    output = model(mock_input)

# Look for flash_attn kernels in profiler output
events = prof.key_averages()
fa_events = [e for e in events if 'flash' in e.key.lower()]
assert len(fa_events) > 0, "FlashAttention not being used — check model config and PyTorch version"

# Method 2: Force and verify
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = model(mock_input)  # Will error if FA not available
```

## MoE Backend

```python
# Verify MoE routing is using optimized kernel
# This is framework-specific — check your MoE implementation
# Key things to verify:
# 1. Expert parallel is enabled (if multi-GPU)
# 2. Token routing uses optimized scatter/gather
# 3. Load balancing auxiliary loss is active
```

## Precision Check

```python
# Verify model is running in expected precision
for name, param in model.named_parameters():
    assert param.dtype == torch.bfloat16, f"{name} is {param.dtype}, expected bf16"

# Check autocast is working
with torch.autocast('cuda', dtype=torch.bfloat16):
    output = model(mock_input)
    assert output.dtype == torch.bfloat16
```

## CUDA Kernel Selection

```python
# Use profiler to check no slow fallback kernels
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    output = model(mock_input)
    loss = criterion(output, mock_target)
    loss.backward()

# Print top CUDA kernels by time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# Look for: unexpected generic kernels instead of optimized ones
```
