---
description: Compare two RAG evaluation runs side-by-side to identify regressions and improvements across metrics
---

# Compare Evaluation Runs

This workflow compares two previously saved evaluation result files to show metric-level diffs (improved / regressed / stable).

## Prerequisites

- The rag_eval venv must be activated
- Two evaluation result JSON files (from previous `rag-eval` runs)

## Steps

1. Activate the rag_eval virtual environment:
// turbo
```bash
source C:/Users/chens34/Downloads/tmp_notes/rag_eval/venv/Scripts/activate
```

2. Compare two runs. Replace placeholders:
   - `<RUN_A_PATH>`: Path to baseline result JSON
   - `<RUN_B_PATH>`: Path to current/candidate result JSON
   - `--label-a` / `--label-b`: Human-readable labels (optional)

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py compare \
    --run-a <RUN_A_PATH> \
    --run-b <RUN_B_PATH> \
    --label-a "Baseline" \
    --label-b "Current"
```

## Example

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py compare \
    --run-a data/results/eval_run_20260506_080415.json \
    --run-b data/results/eval_run_20260507_120000.json
```

## Output

Prints a rich comparison table showing:
- Per-metric score diff (current vs baseline)
- Count of improved / regressed / stable metrics
