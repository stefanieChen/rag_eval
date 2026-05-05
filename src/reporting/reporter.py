"""Reporter: generate JSON and HTML evaluation reports.

Produces structured output files with full evaluation traces
and optional HTML summaries for human review.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.datasets.schema import EvalRunSummary
from src.llm_judge.judge_base import _get_project_root


def save_json_report(
    summary: EvalRunSummary,
    output_dir: Optional[str] = None,
    filename_prefix: str = "eval_run",
) -> str:
    """Save an evaluation run summary as a JSON file.

    Args:
        summary: The evaluation run summary to save.
        output_dir: Directory to save the report. Defaults to data/results/.
        filename_prefix: Prefix for the output filename.

    Returns:
        Path to the saved JSON file.
    """
    if output_dir is None:
        output_dir = str(_get_project_root() / "data" / "results")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    file_path = out_path / filename

    data = summary.model_dump(mode="json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return str(file_path)


def save_html_report(
    summary: EvalRunSummary,
    output_dir: Optional[str] = None,
    filename_prefix: str = "eval_report",
) -> str:
    """Generate and save an HTML evaluation report.

    Args:
        summary: The evaluation run summary.
        output_dir: Directory to save the report.
        filename_prefix: Prefix for the output filename.

    Returns:
        Path to the saved HTML file.
    """
    if output_dir is None:
        output_dir = str(_get_project_root() / "data" / "results")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.html"
    file_path = out_path / filename

    html = _build_html(summary)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(file_path)


def _build_html(summary: EvalRunSummary) -> str:
    """Build HTML report content from an evaluation summary.

    Args:
        summary: Evaluation run summary.

    Returns:
        HTML string.
    """
    scores_rows = ""
    for metric, score in sorted(summary.aggregated_scores.items()):
        bar_width = int(min(score / 5.0, 1.0) * 100) if score > 1 else int(score * 100)
        scores_rows += f"""
        <tr>
            <td>{metric}</td>
            <td>
                <div class="bar-container">
                    <div class="bar" style="width: {bar_width}%">{score:.4f}</div>
                </div>
            </td>
        </tr>"""

    cases_rows = ""
    for i, result in enumerate(summary.per_case_results, 1):
        score_cells = " | ".join(f"{k}: {v:.2f}" for k, v in result.scores.items())
        composite = f"{result.composite_score:.2f}" if result.composite_score else "N/A"
        cases_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{result.question[:80]}...</td>
            <td>{score_cells}</td>
            <td>{composite}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG Evaluation Report — {summary.run_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; color: #333; }}
        h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 0.5rem; }}
        h2 {{ color: #16213e; margin-top: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #16213e; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .bar-container {{ background: #e9ecef; border-radius: 4px; overflow: hidden; }}
        .bar {{ background: linear-gradient(90deg, #e94560, #0f3460); color: white; padding: 2px 8px;
                border-radius: 4px; font-size: 0.85rem; min-width: 60px; text-align: right; }}
        .meta {{ color: #666; font-size: 0.9rem; }}
        .summary-box {{ background: #f0f4f8; border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0; }}
    </style>
</head>
<body>
    <h1>RAG Evaluation Report</h1>
    <div class="summary-box">
        <p class="meta"><strong>Run ID:</strong> {summary.run_id}</p>
        <p class="meta"><strong>Timestamp:</strong> {summary.timestamp}</p>
        <p class="meta"><strong>Test cases:</strong> {summary.num_cases}</p>
        <p class="meta"><strong>Total time:</strong> {summary.total_latency_ms:.0f} ms</p>
    </div>

    <h2>Aggregated Scores</h2>
    <table>
        <tr><th>Metric</th><th>Score</th></tr>
        {scores_rows}
    </table>

    <h2>Per-Case Results</h2>
    <table>
        <tr><th>#</th><th>Question</th><th>Scores</th><th>Composite</th></tr>
        {cases_rows}
    </table>
</body>
</html>"""


def save_experiment_report(
    result: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """Save an A/B experiment result as JSON.

    Args:
        result: ExperimentResult dict.
        output_dir: Directory to save the report.

    Returns:
        Path to the saved file.
    """
    if output_dir is None:
        output_dir = str(_get_project_root() / "data" / "results")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_{timestamp}.json"
    file_path = out_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    return str(file_path)
