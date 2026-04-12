import json
import os
import random
import sys
import time

STEPS = int(os.environ.get("STEPS", 20))
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)

out_dir = os.environ.get("OUT_DIR", "outputs")
os.makedirs(out_dir, exist_ok=True)

for step in range(STEPS):
    loss = 1.0 / (1 + step * 0.1) + random.uniform(-0.01, 0.01)
    print(f"step={step} loss={loss:.4f} step_time=0.01s", flush=True)
    time.sleep(0.01)

# Write fake checkpoint
with open(os.path.join(out_dir, "ckpt.json"), "w") as f:
    json.dump({"final_loss": loss, "steps": STEPS, "seed": SEED}, f)

print("Training done.", flush=True)
