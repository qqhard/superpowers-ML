# Activation Checks (Transformer)

## Attention Distribution

```python
# Hook into attention layers to capture attention weights
attention_weights = {}

def attention_hook(name):
    def hook(module, input, output):
        # output format depends on implementation
        # Common: (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights[name] = output[1].detach()
    return hook

# Register hooks on attention layers
for name, module in model.named_modules():
    if 'attention' in name.lower() or 'attn' in name.lower():
        module.register_forward_hook(attention_hook(name))
```

## Attention Entropy

```python
# High entropy = attention is spread out (usually good early in training)
# Very low entropy = attention collapsed to few positions (potential problem)
import torch.nn.functional as F

for name, weights in attention_weights.items():
    # weights shape: (batch, heads, seq, seq)
    probs = weights.float()
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=torch.float))
    normalized_entropy = entropy / max_entropy

    print(f"{name}: entropy={normalized_entropy:.3f}")
    # Flag if normalized entropy < 0.1 (attention collapsed)
    assert normalized_entropy > 0.1, f"Attention collapsed in {name}"
```
