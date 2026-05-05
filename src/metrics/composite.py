"""Composite scoring: combine multiple metric scores into weighted aggregates.

Allows defining custom weighting schemes to produce a single quality score
from multiple evaluation dimensions.
"""

from typing import Any, Dict, List, Optional


# Default weight presets
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "faithfulness": 0.25,
        "relevancy": 0.25,
        "completeness": 0.20,
        "hallucination_score": 0.15,
        "semantic_similarity": 0.15,
    },
    "faithfulness_focused": {
        "faithfulness": 0.40,
        "relevancy": 0.15,
        "completeness": 0.15,
        "hallucination_score": 0.20,
        "semantic_similarity": 0.10,
    },
    "user_experience": {
        "faithfulness": 0.15,
        "relevancy": 0.35,
        "completeness": 0.30,
        "hallucination_score": 0.10,
        "semantic_similarity": 0.10,
    },
}


def compute_composite_score(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    preset: str = "balanced",
    normalize_to: float = 5.0,
) -> Dict[str, Any]:
    """Compute a weighted composite score from individual metric scores.

    Args:
        scores: Dict of metric_name → score. Scores should be on the
            same scale (e.g., 1-5 for generation, 0-1 for retrieval).
        weights: Custom weight dict. If None, uses preset.
        preset: Name of weight preset ('balanced', 'faithfulness_focused', 'user_experience').
        normalize_to: Normalize the final score to this maximum value.

    Returns:
        Dict with 'composite_score', 'weighted_components', 'weights_used', 'preset'.
    """
    if weights is None:
        weights = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS["balanced"])

    weighted_components: Dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for metric, weight in weights.items():
        if metric in scores:
            # Normalize score to [0, 1] range if on 1-5 scale
            raw_score = scores[metric]
            if raw_score > 1.0:
                normalized = raw_score / normalize_to
            else:
                normalized = raw_score

            weighted_components[metric] = normalized * weight
            weighted_sum += normalized * weight
            total_weight += weight

    # Scale composite to normalize_to range
    if total_weight > 0:
        composite = (weighted_sum / total_weight) * normalize_to
    else:
        composite = 0.0

    return {
        "composite_score": round(composite, 4),
        "weighted_components": weighted_components,
        "weights_used": weights,
        "preset": preset,
        "available_presets": list(WEIGHT_PRESETS.keys()),
    }


def compute_composite_batch(
    per_case_scores: List[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
    preset: str = "balanced",
) -> Dict[str, Any]:
    """Compute composite scores for a batch of test cases.

    Args:
        per_case_scores: List of score dicts, one per test case.
        weights: Custom weight dict.
        preset: Weight preset name.

    Returns:
        Dict with 'per_case' composites, 'mean_composite', and 'std_composite'.
    """
    composites = []
    per_case_results = []

    for scores in per_case_scores:
        result = compute_composite_score(scores, weights=weights, preset=preset)
        composites.append(result["composite_score"])
        per_case_results.append(result)

    import numpy as np
    mean_composite = float(np.mean(composites)) if composites else 0.0
    std_composite = float(np.std(composites)) if composites else 0.0

    return {
        "per_case": per_case_results,
        "mean_composite": round(mean_composite, 4),
        "std_composite": round(std_composite, 4),
        "num_cases": len(per_case_scores),
    }
