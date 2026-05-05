"""Synthetic test set generation via LLM.

Generates question/answer/context triples from source documents,
useful for bootstrapping evaluation when no human-curated test set exists.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm_judge.judge_base import LLMJudge, load_eval_config


_GENERATE_QA_PROMPT = """You are generating evaluation test cases for a RAG (Retrieval-Augmented Generation) system.

Given the following document content, generate {num_questions} diverse question/answer pairs that could be used to test a RAG system's ability to retrieve and synthesize information from this content.

**Document Content:**
{document_content}

**Requirements:**
- Questions should be specific and answerable from the document
- Include a mix of factual, inferential, and comparative questions
- Ground truth answers should be comprehensive but concise
- Context should be the relevant excerpt(s) from the document

Return a JSON object:
```json
{{
    "test_cases": [
        {{
            "question": "...",
            "ground_truth": "...",
            "contexts": ["relevant excerpt 1", "relevant excerpt 2"]
        }}
    ]
}}
```

Return ONLY the JSON object, no other text."""


class TestSetGenerator(LLMJudge):
    """Generate synthetic test sets from source documents using an LLM.

    Reads documents from a directory, sends content to the LLM,
    and generates question/answer/context triples for evaluation.
    """

    def generate_from_text(
        self,
        text: str,
        num_questions: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate test cases from a single text block.

        Args:
            text: Document text content.
            num_questions: Number of Q/A pairs to generate.

        Returns:
            List of test case dicts with question, ground_truth, contexts.
        """
        # Truncate very long texts to avoid token limits
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

        prompt = _GENERATE_QA_PROMPT.format(
            num_questions=num_questions,
            document_content=text,
        )

        raw, _ = self._call_llm(prompt)
        parsed = self._parse_json_response(raw)
        return parsed.get("test_cases", [])

    def generate_from_directory(
        self,
        docs_path: str,
        num_questions_per_doc: int = 3,
        extensions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate test cases from all documents in a directory.

        Args:
            docs_path: Path to the document directory.
            num_questions_per_doc: Questions to generate per document.
            extensions: File extensions to include. Defaults to common text formats.

        Returns:
            List of all generated test case dicts.
        """
        if extensions is None:
            extensions = [".txt", ".md", ".pdf", ".htm", ".html"]

        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        all_cases = []
        for ext in extensions:
            for file_path in docs_dir.rglob(f"*{ext}"):
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    if not text.strip():
                        continue

                    cases = self.generate_from_text(
                        text=text,
                        num_questions=num_questions_per_doc,
                    )
                    # Add source metadata
                    for case in cases:
                        case["metadata"] = {
                            "source_file": str(file_path.relative_to(docs_dir)),
                            "generated": True,
                        }
                    all_cases.extend(cases)
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {e}")

        return all_cases

    def save_test_set(
        self,
        test_cases: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Save generated test cases to a JSON file.

        Args:
            test_cases: List of test case dicts.
            output_path: Path to save the JSON file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(test_cases)} test cases to {output}")
