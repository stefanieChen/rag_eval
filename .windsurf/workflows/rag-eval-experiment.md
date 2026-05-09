---
description: Run an A/B experiment comparing two RAG configurations with statistical significance testing (Wilcoxon/bootstrap/paired-t)
---

# Run A/B Experiment

This workflow compares two RAG configurations on the same test set using pairwise LLM judge comparison and statistical significance tests.

## Prerequisites

- The rag_eval venv must be activated
- Ollama must be running (or OpenAI/Anthropic configured)
- A test set file in JSON/JSONL/CSV format
- A RAG system that supports config overrides (via `RAGClientBase.query_with_config`)

## Steps

1. Activate the rag_eval virtual environment:
// turbo
```bash
source C:/Users/chens34/Downloads/tmp_notes/rag_eval/venv/Scripts/activate
```

2. Run the A/B experiment. Replace the placeholders:
   - `<TEST_SET_PATH>`: Path to the test set
   - `<CONFIG_A_JSON>`: JSON string of config overrides for variant A
   - `<CONFIG_B_JSON>`: JSON string of config overrides for variant B
   - `--label-a` / `--label-b`: Human-readable labels (optional)
   - `--client-type`: RAG client type (`rag2`, `http`, or fully-qualified class path)

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py experiment \
    --test-set <TEST_SET_PATH> \
    --config-a '<CONFIG_A_JSON>' \
    --config-b '<CONFIG_B_JSON>' \
    --label-a "Baseline" \
    --label-b "Candidate"
```

## Example

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py experiment \
    --test-set data/test_sets/sample.json \
    --config-a '{"retrieval.hybrid_mode": true, "retrieval.top_k": 5}' \
    --config-b '{"retrieval.hybrid_mode": false, "retrieval.top_k": 10}' \
    --label-a "Hybrid k=5" \
    --label-b "Dense k=10"
```

## Output

Results are saved to `data/results/experiment_<timestamp>.json` containing:
- `pairwise_results`: Per-question winner (A/B/tie), scores, and reasoning
- `scores_a` / `scores_b`: Aggregated mean, std, win count, win rate
- `statistical_tests`: Test type, p-value, significance flag
- `winner`: Overall winner ("A", "B", or "tie")

## Configuration

Statistical test settings are in `config/eval_config.yaml` under `experiment:`:
- `statistical_test`: `wilcoxon` (default), `bootstrap`, or `paired_t`
- `significance_level`: 0.05 (default)
- `bootstrap_iterations`: 1000 (default)
