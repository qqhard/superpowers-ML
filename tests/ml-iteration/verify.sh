#!/usr/bin/env bash
# Verify that accuracy monotonically improves as STEPS grows
# (sanity check for the test harness, not for the skill itself).
set -euo pipefail

cd "$(dirname "$0")/base-project"

STEPS=20  python3 train.py > /dev/null
acc1=$(python3 evaluate.py | awk -F= '/accuracy/{print $2}')

STEPS=200 python3 train.py > /dev/null
acc2=$(python3 evaluate.py | awk -F= '/accuracy/{print $2}')

python3 -c "
import sys
a1, a2 = float('$acc1'), float('$acc2')
print(f'acc @ STEPS=20: {a1}')
print(f'acc @ STEPS=200: {a2}')
sys.exit(0 if a2 > a1 else 1)
"
