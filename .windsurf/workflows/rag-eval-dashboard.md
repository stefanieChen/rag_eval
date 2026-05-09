---
description: Launch the Streamlit evaluation dashboard to visually explore RAG evaluation results
---

# Launch Evaluation Dashboard

This workflow starts the Streamlit-based dashboard for interactive exploration of evaluation results.

## Prerequisites

- The rag_eval venv must be activated
- Streamlit and plotly installed (`pip install streamlit plotly`)
- At least one evaluation result in `data/results/`

## Steps

1. Activate the rag_eval virtual environment:
// turbo
```bash
source C:/Users/chens34/Downloads/tmp_notes/rag_eval/venv/Scripts/activate
```

2. Launch the dashboard:
```bash
cd C:/Users/chens34/Downloads/tmp_notes/rag_eval && python cli.py dashboard --port 8501
```

3. Open `http://localhost:8501` in your browser.

## Features

- Browse all evaluation runs
- Visualize per-metric score distributions
- Drill down into per-question judge reasoning
- Compare runs side-by-side
