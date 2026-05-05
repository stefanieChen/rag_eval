"""Classical Information Retrieval metrics for RAG retrieval evaluation.

All metrics operate on ranked retrieval results compared against
ground-truth relevant document identifiers or content.
"""

import math
from typing import Dict, List, Optional, Set, Union

import numpy as np


def recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Compute Recall@K — fraction of relevant docs found in top-K results.

    Args:
        retrieved: Ordered list of retrieved document IDs/content.
        relevant: Set of relevant document IDs/content.
        k: Cutoff rank.

    Returns:
        Recall score in [0, 1].
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Compute Precision@K — fraction of top-K results that are relevant.

    Args:
        retrieved: Ordered list of retrieved document IDs/content.
        relevant: Set of relevant document IDs/content.
        k: Cutoff rank.

    Returns:
        Precision score in [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def mrr(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """Compute Mean Reciprocal Rank — 1/rank of the first relevant result.

    Args:
        retrieved: Ordered list of retrieved document IDs/content.
        relevant: Set of relevant document IDs/content.

    Returns:
        MRR score in [0, 1]. Returns 0 if no relevant doc found.
    """
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved: Ordered list of retrieved document IDs/content.
        relevant: Set of relevant document IDs/content.
        k: Cutoff rank.

    Returns:
        NDCG score in [0, 1].
    """
    top_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, doc in enumerate(top_k):
        rel = 1.0 if doc in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG — all relevant docs at top
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """Compute Average Precision for a single query.

    Args:
        retrieved: Ordered list of retrieved document IDs/content.
        relevant: Set of relevant document IDs/content.

    Returns:
        AP score in [0, 1].
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant)


def map_score(
    queries_retrieved: List[List[str]],
    queries_relevant: List[Set[str]],
) -> float:
    """Compute Mean Average Precision across multiple queries.

    Args:
        queries_retrieved: List of retrieved doc lists, one per query.
        queries_relevant: List of relevant doc sets, one per query.

    Returns:
        MAP score in [0, 1].
    """
    if not queries_retrieved:
        return 0.0
    ap_scores = [
        average_precision(ret, rel)
        for ret, rel in zip(queries_retrieved, queries_relevant)
    ]
    return float(np.mean(ap_scores))


def compute_all_retrieval_metrics(
    retrieved: List[str],
    relevant: Set[str],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute all retrieval metrics for a single query.

    Args:
        retrieved: Ordered list of retrieved document IDs/content.
        relevant: Set of relevant document IDs/content.
        k_values: List of K cutoffs for Recall@K, Precision@K, NDCG@K.
            Defaults to [1, 3, 5, 10].

    Returns:
        Dict of metric_name → score.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results: Dict[str, float] = {}

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        results[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)

    results["mrr"] = mrr(retrieved, relevant)
    results["average_precision"] = average_precision(retrieved, relevant)

    return results


def aggregate_retrieval_metrics(
    per_query_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-query retrieval metrics into dataset-level means.

    Args:
        per_query_metrics: List of metric dicts from compute_all_retrieval_metrics.

    Returns:
        Dict of metric_name → mean score across all queries.
    """
    if not per_query_metrics:
        return {}

    all_keys = per_query_metrics[0].keys()
    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in per_query_metrics if key in m]
        aggregated[key] = float(np.mean(values)) if values else 0.0

    return aggregated
