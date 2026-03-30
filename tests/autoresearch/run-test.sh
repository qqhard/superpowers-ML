#!/usr/bin/env bash
# Integration test for autoresearch skill
# Usage: ./run-test.sh [max_rounds]
#
# Sets up a temp project with polynomial fitting base code,
# runs Claude with the autoresearch skill, then verifies results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAX_ROUNDS="${1:-3}"

TIMESTAMP=$(date +%s)
TEST_PROJECT="/tmp/autoresearch-test-${TIMESTAMP}"

echo "=== Autoresearch Integration Test ==="
echo "Test project: $TEST_PROJECT"
echo "Max rounds: $MAX_ROUNDS"
echo ""

# Setup: copy base project and initialize git
mkdir -p "$TEST_PROJECT"
cp "$SCRIPT_DIR/base-project/train.py" "$TEST_PROJECT/"
cp "$SCRIPT_DIR/base-project/evaluate.py" "$TEST_PROJECT/"
cp "$SCRIPT_DIR/base-project/autoresearch-protocol.md" "$TEST_PROJECT/"
cp "$SCRIPT_DIR/base-project/experiences.md" "$TEST_PROJECT/"

# Override max_rounds in protocol if specified
if [ "$MAX_ROUNDS" != "5" ]; then
    sed -i.bak "s/max_rounds: 5/max_rounds: $MAX_ROUNDS/" "$TEST_PROJECT/autoresearch-protocol.md"
    rm -f "$TEST_PROJECT/autoresearch-protocol.md.bak"
fi

cd "$TEST_PROJECT"
git init
git add -A
git commit -m "initial: base code for autoresearch test"

# Verify base code works
echo "Verifying base code..."
python3 train.py --degree 2 --epoch-limit 1 --time-limit 10
python3 evaluate.py
echo ""

# Run Claude with autoresearch skill
PROMPT="I need you to run an automated research loop. Read autoresearch-protocol.md for the protocol, experiences.md for the current state. Use the spml:autoresearch skill."

LOG_FILE="$TEST_PROJECT/claude-output.json"

echo "Running Claude with autoresearch skill..."
echo "Plugin dir: $PLUGIN_DIR"

cd "$PLUGIN_DIR"
timeout 1800 claude -p "$PROMPT" \
    --plugin-dir "$PLUGIN_DIR" \
    --dangerously-skip-permissions \
    --max-turns 100 \
    --add-dir "$TEST_PROJECT" \
    --output-format stream-json \
    > "$LOG_FILE" 2>&1 || true

echo ""
echo "Claude finished. Running verification..."
echo ""

# Run verification
"$SCRIPT_DIR/verify.sh" "$TEST_PROJECT" "$LOG_FILE"
