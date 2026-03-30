#!/usr/bin/env bash
# Verify autoresearch test results
# Usage: ./verify.sh <test-project-dir> <log-file>

set -euo pipefail

TEST_DIR="$1"
LOG_FILE="$2"

PASSED=0
FAILED=0

check() {
    local test_name="$1"
    local result="$2"
    if [ "$result" = "true" ]; then
        echo "  ✅ $test_name"
        PASSED=$((PASSED + 1))
    else
        echo "  ❌ $test_name"
        FAILED=$((FAILED + 1))
    fi
}

echo "=== Verification ==="
echo ""

# 1. Skill invocation
echo "1. Skill invocation"
SKILL_TRIGGERED=$(grep -q '"skill":"autoresearch"' "$LOG_FILE" 2>/dev/null || grep -q '"skill":"spml:autoresearch"' "$LOG_FILE" 2>/dev/null; echo $?)
check "autoresearch skill invoked" "$([ "$SKILL_TRIGGERED" = "0" ] && echo true || echo false)"

# 2. Subagents dispatched
echo "2. Subagent dispatching"
AGENT_COUNT=$(grep -c '"name":"Agent"' "$LOG_FILE" 2>/dev/null || echo "0")
check "subagents dispatched (need ≥2, got $AGENT_COUNT)" "$([ "$AGENT_COUNT" -ge 2 ] && echo true || echo false)"

# 3. Git state
echo "3. Git management"
cd "$TEST_DIR"
COMMIT_COUNT=$(git log --oneline | wc -l | tr -d ' ')
check "multiple git commits ($COMMIT_COUNT total)" "$([ "$COMMIT_COUNT" -gt 1 ] && echo true || echo false)"

# Check for autoresearch commit messages
AR_COMMITS=$(git log --oneline --grep="autoresearch" | wc -l | tr -d ' ')
check "autoresearch commits found ($AR_COMMITS)" "$([ "$AR_COMMITS" -gt 0 ] && echo true || echo false)"

# 4. experiences.md integrity
echo "4. Experience log"
if [ -f "$TEST_DIR/experiences.md" ]; then
    check "experiences.md exists" "true"

    # Check Summary section exists
    HAS_SUMMARY=$(grep -q "## Summary" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "Summary section exists" "$HAS_SUMMARY"

    # Check at least one Round entry
    ROUND_COUNT=$(grep -c "^## Round" "$TEST_DIR/experiences.md" || echo "0")
    check "round entries exist ($ROUND_COUNT)" "$([ "$ROUND_COUNT" -gt 0 ] && echo true || echo false)"

    # Check rounds have required fields
    HAS_STRATEGY=$(grep -q "Strategy" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "rounds have Strategy field" "$HAS_STRATEGY"

    HAS_RESULT=$(grep -q "Result" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "rounds have Result field" "$HAS_RESULT"

    HAS_VERDICT=$(grep -q "Verdict" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "rounds have Verdict field" "$HAS_VERDICT"

    # Check Status is not running (loop terminated)
    STATUS_DONE=$(grep -qE "Status: (completed|target_reached)" "$TEST_DIR/experiences.md" && echo true || echo false)
    check "loop terminated (status not running)" "$STATUS_DONE"
else
    check "experiences.md exists" "false"
fi

# 5. Result quality
echo "5. Result quality"
if [ -f "$TEST_DIR/result.json" ]; then
    check "result.json exists" "true"
else
    check "result.json exists" "false"
fi

# Check final evaluation
FINAL_MSE=$(cd "$TEST_DIR" && python3 evaluate.py 2>/dev/null | sed -n 's/.*mse=\([0-9.]*\).*/\1/p' || echo "N/A")
if [ "$FINAL_MSE" != "N/A" ]; then
    check "final evaluation runs (mse=$FINAL_MSE)" "true"
    # Check if improved from baseline 0.223
    IMPROVED=$(python3 -c "print('true' if float('$FINAL_MSE') < 0.223 else 'false')" 2>/dev/null || echo "false")
    check "improved from baseline 0.223 (got $FINAL_MSE)" "$IMPROVED"
else
    check "final evaluation runs" "false"
fi

echo ""
echo "=== Summary ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "STATUS: FAILED"
    exit 1
else
    echo "STATUS: PASSED"
    exit 0
fi
