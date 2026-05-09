---
description: Generate a synthetic test set from documents using LLM-based question-answer generation for RAG evaluation
---

# Generate Synthetic Test Set

This workflow generates a test set (question + ground truth + contexts) from a directory of documents, using the LLM to synthesize realistic QA pairs.

## Prerequisites

- The rag_eval venv must be activated
- Ollama must be running (or OpenAI/Anthropic configured)
- A directory of source documents (txt, md, pdf, etc.)

## Steps

1. Activate the rag_eval virtual environment:
// turbo
```bash
source C:/Users/chens34/Downloads/tmp_notes/rag_eval/venv/Scripts/activate
```

2. Generate the test set. Replace placeholders:
   - `<DOCS_PATH>`: Path to the documents directory
   - `--num-questions`: Number of questions per document (default: 3)
   - `--output`: Output file path (default: `data/test_sets/generated.json`)

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py generate-testset \
    --docs-path <DOCS_PATH> \
    --num-questions 5 \
    --output data/test_sets/generated.json
```

## Example: Generate from rag_2 documents

```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py generate-testset \
    --docs-path ../rag_2/data/raw/ \
    --num-questions 3
```

## Output

A JSON file containing an array of test cases, each with:
- `question`: Generated question
- `ground_truth`: Expected reference answer
- `contexts`: Source context chunks used to generate the QA pair
- `metadata`: Source document info, difficulty, etc.
