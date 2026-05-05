"""Comparator: compare two evaluation runs and produce diff reports.

Supports statistical comparison of metric differences between runs,
useful for regression detection and progress tracking.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_run(path: str) -> Dict[str, Any]:
    """Load an evaluation run result from a JSON file.

    Args:
        path: Path to the JSON result file.

    Returns:
        Parsed result dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Run result not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_runs(
    run_a: Dict[str, Any],
    run_b: Dict[str, Any],
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> Dict[str, Any]:
    """Compare two evaluation runs and produce a diff report.

    Args:
        run_a: First run result dict.
        run_b: Second run result dict.
        label_a: Label for first run.
        label_b: Label for second run.

    Returns:
        Comparison report dict with deltas, improvements, and regressions.
    """
    scores_a = run_a.get("aggregated_scores", {})
    scores_b = run_b.get("aggregated_scores", {})

    all_metrics = sorted(set(scores_a.keys()) | set(scores_b.keys()))

    comparisons = []
    improvements = []
    regressions = []

    for metric in all_metrics:
        val_a = scores_a.get(metric)
        val_b = scores_b.get(metric)

        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            pct_change = (delta / val_a * 100) if val_a != 0 else 0.0

            entry = {
                "metric": metric,
                label_a: round(val_a, 4),
                label_b: round(val_b, 4),
                "delta": round(delta, 4),
                "pct_change": round(pct_change, 2),
            }

            # Classify change (assuming higher = better for all metrics)
            if delta > 0.01:
                entry["status"] = "improved"
                improvements.append(metric)
            elif delta < -0.01:
                entry["status"] = "regressed"
                regressions.append(metric)
            else:
                entry["status"] = "stable"

            comparisons.append(entry)
        else:
            comparisons.append({
                "metric": metric,
                label_a: val_a,
                label_b: val_b,
                "delta": None,
                "status": "missing_in_one_run",
            })

    return {
        "label_a": label_a,
        "label_b": label_b,
        "run_a_id": run_a.get("run_id", ""),
        "run_b_id": run_b.get("run_id", ""),
        "comparisons": comparisons,
        "improvements": improvements,
        "regressions": regressions,
        "num_improved": len(improvements),
        "num_regressed": len(regressions),
        "num_stable": len(comparisons) - len(improvements) - len(regressions),
    }


def compare_files(
    path_a: str,
    path_b: str,
    label_a: str = "Baseline",
    label_b: str = "Current",
) -> Dict[str, Any]:
    """Compare two evaluation result files.

    Args:
        path_a: Path to first result JSON.
        path_b: Path to second result JSON.
        label_a: Label for first run.
        label_b: Label for second run.

    Returns:
        Comparison report dict.
    """
    run_a = load_run(path_a)
    run_b = load_run(path_b)
    return compare_runs(run_a, run_b, label_a=label_a, label_b=label_b)


def format_comparison_table(comparison: Dict[str, Any]) -> str:
    """Format a comparison report as a text table.

    Args:
        comparison: Comparison report from compare_runs.

    Returns:
        Formatted string table.
    """
    lines = []
    label_a = comparison["label_a"]
    label_b = comparison["label_b"]

    header = f"{'Metric':<30s} {label_a:>12s} {label_b:>12s} {'Delta':>10s} {'Change':>10s} {'Status':>12s}"
    lines.append(header)
    lines.append("-" * len(header))

    for entry in comparison["comparisons"]:
        metric = entry["metric"]
        va = f"{entry[label_a]:.4f}" if entry.get(label_a) is not None else "N/A"
        vb = f"{entry[label_b]:.4f}" if entry.get(label_b) is not None else "N/A"
        delta = f"{entry['delta']:+.4f}" if entry.get("delta") is not None else "N/A"
        pct = f"{entry.get('pct_change', 0):+.1f}%" if entry.get("pct_change") is not None else ""
        status = entry.get("status", "")

        status_icon = {"improved": "+", "regressed": "!", "stable": "=", "missing_in_one_run": "?"}
        icon = status_icon.get(status, " ")

        lines.append(f"{metric:<30s} {va:>12s} {vb:>12s} {delta:>10s} {pct:>10s} {icon:>2s} {status}")

    lines.append("")
    lines.append(f"Improved: {comparison['num_improved']} | "
                 f"Regressed: {comparison['num_regressed']} | "
                 f"Stable: {comparison['num_stable']}")

    return "\n".join(lines)
