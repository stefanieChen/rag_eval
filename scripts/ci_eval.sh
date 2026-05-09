#!/usr/bin/env bash
# CI/CD Evaluation Script for RAG Eval Suite
# ============================================
# Portable script that can be called from any CI system.
#
# Usage:
#   ./scripts/ci_eval.sh                    # Stage 2: quality gate only
#   ./scripts/ci_eval.sh --with-regression  # Stage 2 + 3: quality gate + regression check
#   ./scripts/ci_eval.sh --with-experiment  # Stage 2 + 4: quality gate + A/B experiment
#   ./scripts/ci_eval.sh --all              # Stage 2 + 3 + 4: everything
#
# Environment variables (all optional):
#   RAG_EVAL_TEST_SET     - Path to test set (default: data/test_sets/sample.json)
#   RAG_EVAL_THRESHOLD    - Minimum avg score for quality gate (default: 3.0)
#   RAG_EVAL_BASELINE     - Path to baseline result JSON for regression check
#   RAG_EVAL_MODE         - Evaluation mode: static|pipeline (default: static)
#   RAG_EVAL_CLIENT_TYPE  - RAG client type for pipeline mode (default: rag2)
#   RAG_EVAL_CONFIG_A     - JSON config override for A/B variant A
#   RAG_EVAL_CONFIG_B     - JSON config override for A/B variant B
#   RAG_EVAL_OUTPUT_DIR   - Directory to save results (default: data/results)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
TEST_SET="${RAG_EVAL_TEST_SET:-data/test_sets/sample.json}"
THRESHOLD="${RAG_EVAL_THRESHOLD:-3.0}"
BASELINE="${RAG_EVAL_BASELINE:-}"
MODE="${RAG_EVAL_MODE:-static}"
CLIENT_TYPE="${RAG_EVAL_CLIENT_TYPE:-rag2}"
CONFIG_A="${RAG_EVAL_CONFIG_A:-}"
CONFIG_B="${RAG_EVAL_CONFIG_B:-}"
OUTPUT_DIR="${RAG_EVAL_OUTPUT_DIR:-data/results}"

WITH_REGRESSION=false
WITH_EXPERIMENT=false

for arg in "$@"; do
    case "$arg" in
        --with-regression) WITH_REGRESSION=true ;;
        --with-experiment) WITH_EXPERIMENT=true ;;
        --all) WITH_REGRESSION=true; WITH_EXPERIMENT=true ;;
    esac
done

echo "========================================"
echo " RAG Eval CI/CD Pipeline"
echo "========================================"
echo " Test set:  $TEST_SET"
echo " Mode:      $MODE"
echo " Threshold: $THRESHOLD"
echo "========================================"

# ── Stage 2: Quality Gate ──────────────────────────────────
echo ""
echo "▶ Stage 2: Running evaluation (quality gate)..."

EVAL_CMD="python cli.py eval \
    --test-set $TEST_SET \
    --mode $MODE \
    --output-dir $OUTPUT_DIR \
    --threshold $THRESHOLD"

if [ "$MODE" = "pipeline" ]; then
    EVAL_CMD="$EVAL_CMD --client-type $CLIENT_TYPE"
fi

eval "$EVAL_CMD"
EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "✗ Quality gate FAILED (exit code $EVAL_EXIT)"
    exit 1
fi
echo "✓ Quality gate passed"

# Find the latest result file for subsequent stages
LATEST_RESULT=$(ls -t "$OUTPUT_DIR"/eval_run_*.json 2>/dev/null | head -1)

# ── Stage 3: Regression Check ─────────────────────────────
if [ "$WITH_REGRESSION" = true ]; then
    echo ""
    echo "▶ Stage 3: Regression check..."

    if [ -z "$BASELINE" ]; then
        # Try to find baseline in a well-known location
        BASELINE="$OUTPUT_DIR/baseline.json"
    fi

    if [ ! -f "$BASELINE" ]; then
        echo "⚠ No baseline found at $BASELINE — skipping regression check"
        echo "  To create a baseline, copy a good eval result:"
        echo "  cp $LATEST_RESULT $OUTPUT_DIR/baseline.json"
    else
        python cli.py compare \
            --run-a "$BASELINE" \
            --run-b "$LATEST_RESULT" \
            --label-a "Baseline" \
            --label-b "Current" \
            --fail-on-regression

        if [ $? -ne 0 ]; then
            echo "✗ Regression check FAILED"
            exit 1
        fi
        echo "✓ No regressions detected"
    fi
fi

# ── Stage 4: A/B Experiment ───────────────────────────────
if [ "$WITH_EXPERIMENT" = true ]; then
    echo ""
    echo "▶ Stage 4: A/B experiment..."

    if [ -z "$CONFIG_A" ] || [ -z "$CONFIG_B" ]; then
        echo "⚠ Skipping A/B experiment: RAG_EVAL_CONFIG_A and RAG_EVAL_CONFIG_B not set"
    else
        python cli.py experiment \
            --test-set "$TEST_SET" \
            --config-a "$CONFIG_A" \
            --config-b "$CONFIG_B" \
            --label-a "Variant A" \
            --label-b "Variant B" \
            --output-dir "$OUTPUT_DIR" \
            --client-type "$CLIENT_TYPE"

        echo "✓ A/B experiment completed"
    fi
fi

echo ""
echo "========================================"
echo " CI/CD Pipeline completed successfully"
echo "========================================"
