# RAG Evaluation Suite

A standalone, multi-layer evaluation system for **any** RAG (Retrieval-Augmented Generation) pipeline. Features pluggable RAG client interface, LLM-as-a-Judge, classical IR metrics, semantic similarity, hallucination detection, and A/B experiment comparison.

## Features

- **Multi-layer evaluation**: Retrieval metrics, generation quality, semantic similarity, hallucination detection
- **LLM-as-a-Judge**: Pointwise scoring, pairwise comparison, reference-based grading with configurable rubrics
- **Pluggable LLM providers**: Ollama (local, default), OpenAI, Anthropic — switch via config
- **Pluggable RAG clients**: Evaluate any RAG system — direct import, HTTP API, or custom adapter
- **A/B experiments**: Compare RAG configurations with statistical significance tests
- **Rich reporting**: JSON traces, HTML summaries, Streamlit dashboard
- **CI/CD ready**: Quality gates, regression detection, exit codes for pipeline integration
- **Python SDK**: Programmatic API for embedding eval into your own scripts
- **Windsurf IDE integration**: Slash-command workflows for interactive use

## Architecture

```mermaid
graph TB
    subgraph CLI["CLI  (cli.py)"]
        cmd_eval["eval"]
        cmd_exp["experiment"]
        cmd_gen["generate-testset"]
        cmd_cmp["compare"]
        cmd_rpt["report"]
        cmd_dash["dashboard"]
    end

    subgraph Datasets["Datasets  (src/datasets/)"]
        loader["Loader<br/><i>JSON · JSONL · CSV</i>"]
        schema["Schema<br/><i>TestCase · RAGResponse<br/>EvalTestCase · EvalResult</i>"]
        generator["TestSetGenerator<br/><i>LLM-based synthetic Q&A</i>"]
    end

    subgraph Pipeline["Pipeline  (src/pipeline/)"]
        evaluator["Evaluator<br/><i>Orchestrates full eval runs</i>"]
        client_base["RAGClientBase<br/><i>Abstract interface</i>"]
        rag2_client["Rag2Client<br/><i>Direct rag_2 import</i>"]
        http_client["HttpRAGClient<br/><i>REST API adapter</i>"]
        factory["ClientFactory<br/><i>Config-driven creation</i>"]
        experiment["ExperimentRunner<br/><i>A/B testing + statistics</i>"]
    end

    subgraph Metrics["Metrics  (src/metrics/)"]
        retrieval["Retrieval<br/><i>Recall · Precision · MRR<br/>NDCG · MAP</i>"]
        generation["Generation<br/><i>Faithfulness · Relevancy<br/>Completeness</i>"]
        hallucination["Hallucination<br/><i>Claim decomposition<br/>Grounding check</i>"]
        semantic["Semantic<br/><i>BERTScore<br/>Cosine similarity</i>"]
        composite["Composite<br/><i>Weighted aggregation</i>"]
    end

    subgraph Judge["LLM Judge  (src/llm_judge/)"]
        judge_base["LLMJudge Base<br/><i>LiteLLM · Jinja2 prompts<br/>JSON parsing · Consensus</i>"]
        pointwise["Pointwise<br/><i>1-5 rubric scoring</i>"]
        pairwise["Pairwise<br/><i>A/B + position debiasing</i>"]
        refbased["Reference-Based<br/><i>Grade vs ground truth</i>"]
    end

    subgraph Reporting["Reporting  (src/reporting/)"]
        reporter["Reporter<br/><i>JSON · HTML</i>"]
        comparator["Comparator<br/><i>Run diff analysis</i>"]
        dashboard["Dashboard<br/><i>Streamlit</i>"]
    end

    subgraph Config["Config  (config/)"]
        eval_cfg["eval_config.yaml"]
        rubrics["rubrics/*.yaml"]
    end

    subgraph External["External Systems"]
        rag2["rag_2 Pipeline<br/><i>Direct Python import</i>"]
        any_rag["Any RAG System<br/><i>Via HTTP REST API</i>"]
        llm_providers["LLM Providers<br/><i>Ollama · OpenAI · Anthropic</i>"]
    end

    %% CLI → Pipeline
    cmd_eval --> evaluator
    cmd_exp --> experiment
    cmd_gen --> generator
    cmd_cmp --> comparator
    cmd_rpt --> reporter
    cmd_dash --> dashboard

    %% Pipeline flows
    evaluator --> loader
    evaluator --> factory
    evaluator --> retrieval & generation & hallucination & semantic
    evaluator --> composite
    evaluator --> reporter

    experiment --> loader
    experiment --> factory
    experiment --> pairwise

    %% Factory → Clients
    factory --> client_base
    rag2_client --> client_base
    http_client --> client_base

    %% Clients → External
    rag2_client --> rag2
    http_client --> any_rag

    %% Metrics → Judge
    generation --> judge_base
    hallucination --> judge_base

    %% Judge hierarchy
    pointwise --> judge_base
    pairwise --> judge_base
    refbased --> judge_base

    %% Judge → LLM
    judge_base --> llm_providers

    %% Generator
    generator --> judge_base

    %% Config
    eval_cfg -.-> evaluator
    eval_cfg -.-> judge_base
    rubrics -.-> judge_base

    %% Styling
    classDef external fill:#f9f0ff,stroke:#7c3aed,stroke-width:2px
    classDef pipeline fill:#eff6ff,stroke:#2563eb,stroke-width:2px
    classDef metrics fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    classDef judge fill:#fff7ed,stroke:#ea580c,stroke-width:2px
    classDef report fill:#fdf2f8,stroke:#db2777,stroke-width:2px
    classDef config fill:#f5f5f4,stroke:#78716c,stroke-width:1px,stroke-dasharray: 5 5

    class rag2,any_rag,llm_providers external
    class evaluator,client_base,rag2_client,http_client,factory,experiment pipeline
    class retrieval,generation,hallucination,semantic,composite metrics
    class judge_base,pointwise,pairwise,refbased judge
    class reporter,comparator,dashboard report
    class eval_cfg,rubrics config
```

**Data flow summary:**

1. **CLI** dispatches commands to pipeline orchestrators
2. **Evaluator** loads test sets → queries RAG (pipeline mode) or uses static contexts → scores with all metric layers → produces reports
3. **ExperimentRunner** queries two RAG configs per question → pairwise LLM judge comparison → statistical significance tests (Wilcoxon / bootstrap / paired-t)
4. **Metrics** layer fans out to classical IR, LLM-judged generation quality, claim-level hallucination detection, and embedding-based semantic similarity
5. **LLM Judge** calls are unified through LiteLLM (Ollama/OpenAI/Anthropic) with Jinja2 prompt templates and optional multi-judge consensus
6. **Reporting** produces JSON traces, HTML summaries, run-vs-run diffs, and a Streamlit dashboard

## Quick Start

```bash
# 1. Create venv and install dependencies
python -m venv venv
source venv/Scripts/activate  # Windows (git bash)
# source venv/bin/activate    # Linux/macOS
pip install -r requirements.txt

# 2. Ensure Ollama is running (for local LLM judge)
ollama pull qwen2.5:7b

# 3. Run evaluation on a test set
python cli.py eval --test-set data/test_sets/sample.json

# 4. Run specific metrics
python cli.py eval --test-set data/test_sets/sample.json --metrics retrieval,faithfulness

# 5. Run with CI quality gate (exit 1 if avg score < threshold)
python cli.py eval --test-set data/test_sets/sample.json --threshold 3.0

# 6. A/B experiment (compare two LLM models)
python cli.py experiment --test-set data/test_sets/sample.json \
    --config-a '{"llm.model": "qwen2.5:7b"}' \
    --config-b '{"llm.model": "qwen3:8b"}' \
    --client-type rag2

# 7. Generate synthetic test set
python cli.py generate-testset --docs-path ../rag_2/data/raw/ --num-questions 50

# 8. Compare runs (with regression gate)
python cli.py compare --run-a data/results/baseline.json --run-b data/results/current.json \
    --fail-on-regression

# 9. Launch dashboard
python cli.py dashboard
```

## Evaluation Layers

### 1. Retrieval Metrics (`src/metrics/retrieval.py`)
- **Recall@K**, **Precision@K**, **MRR**, **NDCG**, **MAP**
- **Context Relevance** (LLM judge): per-chunk relevance scoring
- **Context Sufficiency** (LLM judge): collective completeness check

### 2. Generation Metrics (`src/metrics/generation.py`)
- **Faithfulness**: claim decomposition → grounding verification
- **Answer Relevancy**: question-answer alignment
- **Completeness**: coverage of ground truth
- **Conciseness**: appropriate brevity

### 3. Hallucination Detection (`src/metrics/hallucination.py`)
- Claim-level decomposition
- Per-claim grounding check against retrieved context
- Hallucination rate calculation

### 4. Semantic Metrics (`src/metrics/semantic.py`)
- Embedding cosine similarity (answer vs ground truth)
- BERTScore (token-level F1)
- Fast, no LLM call needed — ideal for regression checks

### 5. LLM-as-a-Judge (`src/llm_judge/`)

| Mode | Use Case | Method |
|------|----------|--------|
| **Pointwise** | Score a single response (1-5 scale) | Rubric-guided prompt |
| **Pairwise** | A/B comparison | "Which is better and why?" |
| **Reference-based** | Grade vs ground truth | Structured comparison |

### 6. A/B Experiments (`src/pipeline/experiment.py`)
- Multi-config comparison on identical test sets
- Pairwise LLM judge + statistical significance (Wilcoxon / bootstrap)

## Connecting to Your RAG System

The evaluation suite uses a **pluggable client interface** (`RAGClientBase`). Three ways to connect:

### Option 1: HTTP API (recommended for most systems)

If your RAG system has a REST API:

```yaml
# config/eval_config.yaml
rag:
  client_type: "http"
  http:
    base_url: "http://localhost:8000"
    query_endpoint: "/query"
    method: "POST"
    request_body_template:
      question: "{question}"
    response_mapping:
      answer: "answer"                # dot-path into JSON response
      contexts: "sources[].content"   # supports []-expansion
      context_ids: "sources[].id"
```

Or via CLI: `python cli.py eval --test-set data/test_sets/sample.json --mode pipeline --client-type http`

### Option 2: Direct import (rag_2)

```yaml
rag:
  client_type: "rag2"
  project_path: "../rag_2"
  config_path: "../rag_2/config/settings.yaml"
```

### Option 3: Custom client (Python)

Implement `RAGClientBase` for any system:

```python
from src.pipeline.client_base import RAGClientBase
from src.datasets.schema import RAGResponse

class MyRAGClient(RAGClientBase):
    def query(self, question: str) -> RAGResponse:
        result = my_rag_system.ask(question)
        return RAGResponse(
            answer=result["answer"],
            retrieved_contexts=result["contexts"],
        )
```

Use it via config (`client_type: "my_package.MyRAGClient"`) or inject directly:

```python
from src.pipeline.evaluator import Evaluator
evaluator = Evaluator(rag_client=MyRAGClient())
summary = evaluator.run("test_set.json", mode="pipeline")
```

## Configuration

Edit `config/eval_config.yaml`:

```yaml
judge:
  provider: "ollama"        # "ollama" | "openai" | "anthropic"
  model: "qwen2.5:7b"
  temperature: 0.0

rag:
  client_type: "rag2"       # "rag2" | "http" | fully-qualified class path
  project_path: "../rag_2"
  config_path: "../rag_2/config/settings.yaml"
```

Rubrics are defined in `config/rubrics/*.yaml` — human-readable scoring guides.

## Project Structure

```
rag_eval/
├── .github/workflows/         # CI/CD pipeline (GitHub Actions)
│   └── rag-eval-ci.yml        # Multi-stage eval + regression + A/B
├── .windsurf/                 # Windsurf IDE integration
│   ├── rules/                 # Skill declarations
│   └── workflows/             # Slash-command workflows
├── config/
│   ├── eval_config.yaml       # Main configuration
│   └── rubrics/               # LLM judge rubric definitions
├── scripts/
│   └── ci_eval.sh             # Portable CI/CD evaluation script
├── src/
│   ├── __init__.py            # SDK public API exports
│   ├── llm_judge/             # LLM-as-a-Judge core
│   ├── metrics/               # All metric implementations
│   ├── pipeline/              # Evaluation orchestration
│   │   ├── client_base.py     # RAGClientBase abstract interface
│   │   ├── rag_client.py      # Rag2Client (direct rag_2 import)
│   │   ├── http_client.py     # HttpRAGClient (REST API adapter)
│   │   ├── client_factory.py  # Config-driven client creation
│   │   ├── evaluator.py       # Evaluation orchestrator
│   │   └── experiment.py      # A/B experiment runner
│   ├── datasets/              # Test set management & generation
│   └── reporting/             # Results, reports, dashboard
├── data/
│   ├── test_sets/             # Test datasets
│   └── results/               # Evaluation outputs
├── tests/                     # Unit tests
├── cli.py                     # Unified CLI
├── requirements.txt
└── README.md
```

## CI/CD Integration

The suite is designed for CI/CD pipeline integration with quality gates and regression detection.

### Quick Setup

```bash
# Run eval with quality gate (exit code 1 if avg score < 3.0)
python cli.py eval --test-set data/test_sets/sample.json --threshold 3.0

# Compare against baseline (exit code 1 if any metric regressed)
python cli.py compare --run-a baseline.json --run-b current.json --fail-on-regression
```

### CI Pipeline Stages

| Stage | Trigger | Mode | Time | Purpose |
|-------|---------|------|------|---------|
| **Unit Tests** | Every PR | — | ~1m | `pytest` basic tests |
| **Quality Gate** | Every PR | static | ~5m | Eval + score threshold check |
| **Regression Check** | PR only | — | ~1m | Compare current vs main baseline |
| **A/B Experiment** | Manual/nightly | pipeline | ~30m | Compare two RAG configurations |

### Portable Script

```bash
# Quality gate only
./scripts/ci_eval.sh

# Quality gate + regression check
./scripts/ci_eval.sh --with-regression

# Quality gate + A/B experiment
RAG_EVAL_CONFIG_A='{"llm.model": "qwen2.5:7b"}' \
RAG_EVAL_CONFIG_B='{"llm.model": "qwen3:8b"}' \
./scripts/ci_eval.sh --with-experiment

# Everything
./scripts/ci_eval.sh --all
```

Environment variables: `RAG_EVAL_THRESHOLD`, `RAG_EVAL_TEST_SET`, `RAG_EVAL_MODE`, `RAG_EVAL_BASELINE`, `RAG_EVAL_CONFIG_A/B`.

### GitHub Actions

A ready-to-use workflow is provided at `.github/workflows/rag-eval-ci.yml`. It requires a **self-hosted runner** with Ollama installed for the LLM judge. The A/B experiment stage supports manual trigger with custom config inputs.

> **Note**: For teams without a self-hosted runner, use `static` mode with a remote LLM provider (e.g., OpenAI) by setting `judge.provider: "openai"` in `eval_config.yaml`.

## Python SDK

All core components are importable for programmatic use:

```python
from src import Evaluator, ExperimentRunner, RAGClientBase
from src import TestCase, RAGResponse, EvalTestCase, EvalResult, EvalRunSummary
from src import load_eval_config, load_rubric

# Run evaluation programmatically
evaluator = Evaluator()
summary = evaluator.run("data/test_sets/sample.json", mode="static")
print(summary.aggregated_scores)

# A/B experiment
runner = ExperimentRunner()
result = runner.run(
    test_set_path="data/test_sets/sample.json",
    config_a={"llm.model": "qwen2.5:7b"},
    config_b={"llm.model": "qwen3:8b"},
)
print(result.winner, result.statistical_tests)
```

Config loading supports environment variables for flexible deployment:
- `RAG_EVAL_CONFIG` — path to `eval_config.yaml`
- `RAG_EVAL_RUBRICS_DIR` — path to rubrics directory

## Windsurf IDE Integration

For [Windsurf](https://codeium.com/windsurf) users, slash-command workflows are available:

| Command | Description |
|---------|-------------|
| `/rag-eval` | Run evaluation on a test set |
| `/rag-eval-experiment` | Run A/B experiment with statistical testing |
| `/rag-eval-compare` | Compare two evaluation runs |
| `/rag-eval-generate-testset` | Generate synthetic test set from documents |
| `/rag-eval-sdk` | Programmatic Python SDK usage guide |
| `/rag-eval-dashboard` | Launch Streamlit evaluation dashboard |

Type any slash command in the Cascade chat to trigger the corresponding workflow.

## Testing

```bash
python -m pytest tests/ -v
```
