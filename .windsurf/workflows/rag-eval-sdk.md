---
description: Use the rag_eval Python SDK programmatically to evaluate a RAG system from code (not CLI)
---

# Use RAG Eval Python SDK

This workflow shows how to call the rag_eval evaluation suite programmatically from Python code, useful for integration into scripts, CI pipelines, or other projects.

## Prerequisites

- The rag_eval project is accessible on the Python path
- Dependencies installed (`pip install -r requirements.txt`)
- Ollama running (or OpenAI/Anthropic configured via env vars)

## Step 1: Basic Evaluation (Static Mode)

Run evaluation against a test set using pre-defined contexts (no live RAG needed):

```python
import sys
sys.path.insert(0, "C:/Users/chens34/Downloads/tmp_notes/rag_eval")

from src import Evaluator

evaluator = Evaluator()
summary = evaluator.run(
    test_set_path="C:/Users/chens34/Downloads/tmp_notes/rag_eval/data/test_sets/sample.json",
    mode="static",
    metrics=["faithfulness", "relevancy", "semantic_similarity"],
)

print(f"Cases: {summary.num_cases}")
print(f"Scores: {summary.aggregated_scores}")
print(f"Latency: {summary.total_latency_ms:.0f}ms")
```

## Step 2: Pipeline Mode with Custom RAG Client

Evaluate a live RAG system by implementing `RAGClientBase`:

```python
import sys
sys.path.insert(0, "C:/Users/chens34/Downloads/tmp_notes/rag_eval")

from src import Evaluator, RAGClientBase, RAGResponse

class MyRAGClient(RAGClientBase):
    def query(self, question: str) -> RAGResponse:
        # Replace with your actual RAG system call
        result = my_rag_system.ask(question)
        return RAGResponse(
            answer=result["answer"],
            retrieved_contexts=result["contexts"],
        )

evaluator = Evaluator(rag_client=MyRAGClient())
summary = evaluator.run("test_set.json", mode="pipeline")
```

## Step 3: A/B Experiment from Code

```python
from src import ExperimentRunner

runner = ExperimentRunner(rag_client=my_client)
result = runner.run(
    test_set_path="test_set.json",
    config_a={"retrieval.hybrid_mode": True},
    config_b={"retrieval.hybrid_mode": False},
    label_a="Hybrid",
    label_b="Dense",
)

print(f"Winner: {result.winner}")
print(f"p-value: {result.statistical_tests['p_value']}")
```

## Step 4: Custom Config Path

If using rag_eval from another project, point to the config explicitly:

```python
import os
os.environ["RAG_EVAL_CONFIG"] = "C:/Users/chens34/Downloads/tmp_notes/rag_eval/config/eval_config.yaml"
os.environ["RAG_EVAL_RUBRICS_DIR"] = "C:/Users/chens34/Downloads/tmp_notes/rag_eval/config/rubrics"

from src import Evaluator
evaluator = Evaluator()
```

Or pass config directly:

```python
from src import load_eval_config, Evaluator

config = load_eval_config("C:/path/to/my/eval_config.yaml")
evaluator = Evaluator(config=config)
```

## Key Classes Reference

| Class | Import | Purpose |
|-------|--------|---------|
| `Evaluator` | `from src import Evaluator` | Run full evaluation on test set |
| `ExperimentRunner` | `from src import ExperimentRunner` | A/B experiment with stats |
| `RAGClientBase` | `from src import RAGClientBase` | Abstract interface to implement |
| `RAGResponse` | `from src import RAGResponse` | Structured RAG query response |
| `TestCase` | `from src import TestCase` | Single test case schema |
| `EvalRunSummary` | `from src import EvalRunSummary` | Full evaluation run result |
| `load_eval_config` | `from src import load_eval_config` | Load config from file/env |
