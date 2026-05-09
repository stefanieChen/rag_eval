---
trigger: always
---

# RAG Evaluation Suite Skill

You have access to a **RAG Evaluation Suite** located at `C:/Users/chens34/Downloads/tmp_notes/rag_eval`. This is a multi-layer evaluation system for any RAG pipeline.

## Capabilities

You can perform the following evaluation tasks:

1. **Run evaluation** (`/rag-eval`): Score a RAG system across 6 metric layers — retrieval (Recall/Precision/MRR/NDCG/MAP), faithfulness, relevancy, completeness, hallucination detection, and semantic similarity. Supports static mode (test set contexts) and pipeline mode (live RAG queries).

2. **A/B experiment** (`/rag-eval-experiment`): Compare two RAG configurations on the same test set with pairwise LLM judge comparison and statistical significance tests (Wilcoxon / bootstrap / paired-t).

3. **Generate test set** (`/rag-eval-generate-testset`): Auto-generate synthetic QA test sets from a directory of documents using LLM-based question generation.

4. **Compare runs** (`/rag-eval-compare`): Diff two evaluation result files to identify metric regressions and improvements.

5. **View dashboard** (`/rag-eval-dashboard`): Launch a Streamlit dashboard for interactive result exploration.

6. **Programmatic SDK** (`/rag-eval-sdk`): Use the Python API directly from code — `from src import Evaluator, RAGClientBase`.

## When to Use

- When the user asks to **evaluate a RAG system**, **measure RAG quality**, or **score retrieval/generation performance** → use `/rag-eval`
- When the user wants to **compare two RAG configs** or **A/B test** → use `/rag-eval-experiment`
- When the user needs a **test set** or **benchmark data** for RAG evaluation → use `/rag-eval-generate-testset`
- When the user asks to **compare evaluation runs** or check for **regressions** → use `/rag-eval-compare`
- When the user wants to **write Python code** that calls the evaluation suite → use `/rag-eval-sdk`

## Key Architecture

- **CLI entry point**: `cli.py` (Click-based, commands: `eval`, `experiment`, `generate-testset`, `compare`, `report`, `dashboard`)
- **Core orchestrator**: `src/pipeline/evaluator.py` → `Evaluator` class
- **A/B runner**: `src/pipeline/experiment.py` → `ExperimentRunner` class
- **Pluggable RAG interface**: `src/pipeline/client_base.py` → `RAGClientBase` (abstract); built-in clients: `rag2`, `http`
- **LLM Judge**: `src/llm_judge/judge_base.py` → LiteLLM-based, supports Ollama/OpenAI/Anthropic
- **Config**: `config/eval_config.yaml` + `config/rubrics/*.yaml`
- **SDK exports**: `from src import Evaluator, ExperimentRunner, RAGClientBase, RAGResponse, TestCase, EvalRunSummary, load_eval_config`

## Environment

- **Venv**: `source C:/Users/chens34/Downloads/tmp_notes/rag_eval/venv/Scripts/activate`
- **Config override**: Set `RAG_EVAL_CONFIG` env var to point to a custom `eval_config.yaml`
- **Rubrics override**: Set `RAG_EVAL_RUBRICS_DIR` env var to point to a custom rubrics directory
- **Default LLM**: Ollama with `qwen2.5:7b` (local, no API key needed)
