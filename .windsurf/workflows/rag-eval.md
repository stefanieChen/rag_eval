---
description: Run RAG evaluation on a test set using the rag_eval suite (multi-layer metrics including retrieval, faithfulness, relevancy, completeness, hallucination, semantic similarity)
---

# Run RAG Evaluation

This workflow runs the RAG Evaluation Suite against a test set file and produces evaluation results.

## Prerequisites

- The rag_eval venv must be activated
- Ollama must be running with the configured model (default: `qwen2.5:7b`)
- A test set file in JSON, JSONL, or CSV format

## Steps

1. Activate the rag_eval virtual environment:
// turbo
```bash
source C:/Users/chens34/Downloads/tmp_notes/rag_eval/venv/Scripts/activate
```

2. Verify Ollama is accessible (skip if using OpenAI/Anthropic):
```bash
curl -s http://localhost:11434/api/tags | python -c "import sys,json; tags=json.load(sys.stdin); print([m['name'] for m in tags.get('models',[])])"
```

3. Run the evaluation. Replace `<TEST_SET_PATH>` with the actual path to the test set file. Available options:
   - `--mode static` (default): Uses contexts from the test set, no live RAG query
   - `--mode pipeline`: Queries a live RAG system
   - `--metrics`: Comma-separated list (e.g., `retrieval,faithfulness,relevancy,completeness,hallucination,semantic_similarity`). Default: all
   - `--format`: `json` (default), `html`, or `both`
   - `--output-dir`: Directory for results (default: `data/results/`)
   - `--client-type`: RAG client type for pipeline mode (`rag2`, `http`, or fully-qualified class path)

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py eval --test-set <TEST_SET_PATH>
```

4. View the latest evaluation report:
// turbo
```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py report
```

## Example: Run with specific metrics

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py eval \
    --test-set data/test_sets/sample.json \
    --metrics retrieval,faithfulness \
    --format both
```

## Example: Pipeline mode with HTTP client

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py eval \
    --test-set data/test_sets/sample.json \
    --mode pipeline \
    --client-type http
```

## Output

Results are saved to `data/results/eval_run_<timestamp>.json` by default. The JSON contains:
- `aggregated_scores`: Mean scores per metric across all test cases
- `per_case_results`: Detailed per-question scores, judge reasoning, and hallucination analysis
- `config_snapshot`: The config used for this run
- `total_latency_ms`: Total evaluation time
