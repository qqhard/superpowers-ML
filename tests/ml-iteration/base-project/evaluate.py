import json
import os
import sys

ckpt_path = os.environ.get("CKPT", "outputs/ckpt.json")

with open(ckpt_path) as f:
    ckpt = json.load(f)

# Deterministic "accuracy" from final_loss — monotonic: lower loss → higher accuracy
accuracy = max(0.0, 1.0 - ckpt["final_loss"])
print(f"accuracy={accuracy:.4f}")
print(f"duration_s=0.2")
