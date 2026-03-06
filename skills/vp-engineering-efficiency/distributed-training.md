# Distributed Training Checks

Only load this when training on multi-GPU or multi-node.

## NCCL Bandwidth

```python
import torch
import torch.distributed as dist
import time

def measure_nccl_bandwidth(tensor_size_mb=100):
    """Measure all-reduce bandwidth"""
    tensor = torch.randn(tensor_size_mb * 1024 * 1024 // 4, device='cuda')

    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    n_iters = 20
    for _ in range(n_iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Bus bandwidth = data_size * 2 * (n-1) / n / time (for ring all-reduce)
    world_size = dist.get_world_size()
    data_bytes = tensor.nelement() * tensor.element_size()
    bus_bw = data_bytes * 2 * (world_size - 1) / world_size / (elapsed / n_iters) / 1e9
    print(f"NCCL bus bandwidth: {bus_bw:.1f} GB/s")
    return bus_bw

# Expected: ~300 GB/s for NVLink, ~12 GB/s for PCIe 4.0 x16
```

## Communication/Computation Overlap

```python
# Check if backward pass overlaps with gradient all-reduce
# Use torch profiler and look for overlap between NCCL kernels and compute kernels
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    output = model(mock_input)
    loss = criterion(output, mock_target)
    loss.backward()
    optimizer.step()

# Export to Chrome trace for visual inspection
prof.export_chrome_trace("trace.json")
# Look for: NCCL all-reduce running concurrently with backward compute
```

## Gradient Synchronization

```python
# Verify gradients are correctly synchronized across ranks
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_sum = param.grad.clone()
        dist.all_reduce(grad_sum)
        expected = param.grad * dist.get_world_size()
        assert torch.allclose(grad_sum, expected, rtol=1e-3), \
            f"Gradient sync issue in {name}"
```
