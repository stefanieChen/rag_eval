"""Test set loader: load and validate test sets from JSON, JSONL, or CSV files."""

import csv
import json
from pathlib import Path
from typing import List, Union

from src.datasets.schema import TestCase


def load_test_set(path: Union[str, Path]) -> List[TestCase]:
    """Load a test set from a file (JSON, JSONL, or CSV).

    JSON: expects a list of objects with 'question', 'ground_truth', 'contexts'.
    JSONL: one JSON object per line.
    CSV: columns 'question', 'ground_truth', 'contexts' (contexts as semicolon-separated).

    Args:
        path: Path to the test set file.

    Returns:
        List of validated TestCase objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or data is invalid.
    """
    path = Path(path)

    suffix = path.suffix.lower()
    if suffix not in (".json", ".jsonl", ".csv"):
        raise ValueError(f"Unsupported test set format: {suffix}. Use .json, .jsonl, or .csv")

    if not path.exists():
        raise FileNotFoundError(f"Test set file not found: {path}")

    if suffix == ".json":
        return _load_json(path)
    elif suffix == ".jsonl":
        return _load_jsonl(path)
    elif suffix == ".csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported test set format: {suffix}. Use .json, .jsonl, or .csv")


def _load_json(path: Path) -> List[TestCase]:
    """Load test cases from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of TestCase objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON test set must be a list, got {type(data).__name__}")

    return [_parse_test_case(item) for item in data]


def _load_jsonl(path: Path) -> List[TestCase]:
    """Load test cases from a JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file.

    Returns:
        List of TestCase objects.
    """
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                cases.append(_parse_test_case(item))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    return cases


def _load_csv(path: Path) -> List[TestCase]:
    """Load test cases from a CSV file.

    Expected columns: question, ground_truth, contexts (semicolon-separated).

    Args:
        path: Path to CSV file.

    Returns:
        List of TestCase objects.
    """
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contexts_raw = row.get("contexts", "")
            contexts = [c.strip() for c in contexts_raw.split(";") if c.strip()]
            cases.append(TestCase(
                question=row.get("question", ""),
                ground_truth=row.get("ground_truth", ""),
                contexts=contexts,
            ))
    return cases


def _parse_test_case(item: dict) -> TestCase:
    """Parse a dict into a TestCase, handling various field name conventions.

    Args:
        item: Raw dict from JSON/JSONL.

    Returns:
        Validated TestCase.
    """
    question = item.get("question", item.get("query", ""))
    ground_truth = item.get("ground_truth", item.get("expected_answer", item.get("reference", "")))
    contexts = item.get("contexts", item.get("context", []))
    if isinstance(contexts, str):
        contexts = [contexts]
    metadata = item.get("metadata", {})

    return TestCase(
        question=question,
        ground_truth=ground_truth,
        contexts=contexts,
        metadata=metadata,
    )
