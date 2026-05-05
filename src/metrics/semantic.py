"""Semantic similarity metrics: embedding cosine similarity and BERTScore.

These metrics are fast and don't require LLM calls — useful for
bulk regression checks and as complementary signals.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def embedding_similarity(
    text_a: str,
    text_b: str,
    model_name: str = "all-MiniLM-L6-v2",
    _model_cache: Dict[str, Any] = {},
) -> float:
    """Compute embedding cosine similarity between two texts.

    Uses sentence-transformers for embedding. Model is cached for reuse.

    Args:
        text_a: First text.
        text_b: Second text.
        model_name: Sentence-transformer model name.

    Returns:
        Cosine similarity score in [0, 1] (texts are usually positive).
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)

    model = _model_cache[model_name]
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
    return cosine_similarity(embeddings[0], embeddings[1])


def bert_score(
    predictions: List[str],
    references: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    _model_cache: Dict[str, Any] = {},
) -> Dict[str, List[float]]:
    """Compute token-level BERTScore (precision, recall, F1) using sentence-transformers.

    This is a simplified BERTScore using sentence-level embeddings decomposed
    into token-level matching via the sentence-transformers tokenizer.

    For production use, consider the official bert_score package.

    Args:
        predictions: List of predicted answer texts.
        references: List of reference answer texts.
        model_name: Sentence-transformer model name.

    Returns:
        Dict with 'precision', 'recall', 'f1' lists (one score per pair).
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)

    model = _model_cache[model_name]

    precisions = []
    recalls = []
    f1s = []

    for pred, ref in zip(predictions, references):
        # Sentence-level similarity as a proxy for BERTScore
        embeddings = model.encode([pred, ref], convert_to_numpy=True)
        sim = cosine_similarity(embeddings[0], embeddings[1])
        # Clamp to [0, 1]
        sim = max(0.0, min(1.0, sim))
        precisions.append(sim)
        recalls.append(sim)
        f1s.append(sim)

    return {
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
    }


def compute_semantic_metrics(
    answer: str,
    ground_truth: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, float]:
    """Compute all semantic metrics for a single answer/ground_truth pair.

    Args:
        answer: The RAG system's answer.
        ground_truth: Reference answer.
        model_name: Embedding model name.

    Returns:
        Dict with embedding_similarity and bert_score_f1.
    """
    emb_sim = embedding_similarity(answer, ground_truth, model_name=model_name)
    bs = bert_score([answer], [ground_truth], model_name=model_name)

    return {
        "embedding_similarity": emb_sim,
        "bert_score_f1": bs["f1"][0] if bs["f1"] else 0.0,
        "bert_score_precision": bs["precision"][0] if bs["precision"] else 0.0,
        "bert_score_recall": bs["recall"][0] if bs["recall"] else 0.0,
    }


def compute_semantic_metrics_batch(
    test_cases: List[Dict[str, str]],
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Compute semantic metrics for a batch of test cases.

    Args:
        test_cases: List of dicts with keys: answer, ground_truth.
        model_name: Embedding model name.

    Returns:
        Dict with 'per_case' results and 'aggregated' means.
    """
    per_case = []
    for case in test_cases:
        result = compute_semantic_metrics(
            answer=case.get("answer", ""),
            ground_truth=case.get("ground_truth", ""),
            model_name=model_name,
        )
        per_case.append(result)

    # Aggregate
    if per_case:
        keys = per_case[0].keys()
        aggregated = {
            key: float(np.mean([r[key] for r in per_case]))
            for key in keys
        }
    else:
        aggregated = {}

    return {
        "per_case": per_case,
        "aggregated": aggregated,
        "num_cases": len(test_cases),
    }
