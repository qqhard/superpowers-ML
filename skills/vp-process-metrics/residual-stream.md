# Residual Stream Checks

## Write Ratio

The residual stream write ratio measures how much each layer modifies the residual stream relative to the stream's magnitude.

```python
residual_norms = {}

def residual_hook(name):
    def hook(module, input, output):
        if isinstance(input, tuple):
            residual_in = input[0]
        else:
            residual_in = input

        residual_out = output if not isinstance(output, tuple) else output[0]

        # Write = how much this layer changed the residual
        write = (residual_out - residual_in).norm()
        stream_norm = residual_in.norm()
        write_ratio = (write / (stream_norm + 1e-10)).item()

        residual_norms[name] = write_ratio
    return hook

# Register on residual-producing layers (e.g., transformer blocks)
for name, module in model.named_modules():
    if 'block' in name.lower() or 'layer' in name.lower():
        module.register_forward_hook(residual_hook(name))

# After forward pass:
for name, ratio in residual_norms.items():
    print(f"{name}: write_ratio={ratio:.4f}")
    # Flag if write ratio > 1.0 (layer output larger than residual = potential instability)
    # Flag if write ratio < 1e-4 (layer barely contributing)
```
