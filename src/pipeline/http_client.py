"""HTTP-based RAG client for evaluating any RAG system via its REST API.

Connects to any RAG system that exposes an HTTP endpoint, making the
evaluation framework completely RAG-implementation agnostic.
"""

import time
from typing import Any, Dict, List, Optional

import requests

from src.datasets.schema import RAGResponse
from src.llm_judge.judge_base import load_eval_config
from src.logging import get_logger
from src.pipeline.client_base import RAGClientBase


class HttpRAGClient(RAGClientBase):
    """RAG client that communicates with any RAG system over HTTP.

    The target RAG system must expose a query endpoint that accepts a JSON
    body and returns a JSON response.  Field mappings are configurable so
    that virtually any API shape can be adapted.

    Minimal config example (in eval_config.yaml)::

        rag:
          client_type: "http"
          http:
            base_url: "http://localhost:8000"
            query_endpoint: "/query"
            method: "POST"
            request_body_template:
              question: "{question}"
            response_mapping:
              answer: "answer"
              contexts: "sources[].content"
              context_ids: "sources[].id"
            headers:
              Content-Type: "application/json"
            timeout: 30
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_eval_config()

        http_cfg = config.get("rag", {}).get("http", {})
        self._base_url = http_cfg.get("base_url", "http://localhost:8000").rstrip("/")
        self._query_endpoint = http_cfg.get("query_endpoint", "/query")
        self._method = http_cfg.get("method", "POST").upper()
        self._headers = http_cfg.get("headers", {"Content-Type": "application/json"})
        self._timeout = http_cfg.get("timeout", 30)
        self._request_template = http_cfg.get("request_body_template", {"question": "{question}"})
        self._response_mapping = http_cfg.get("response_mapping", {
            "answer": "answer",
            "contexts": "contexts",
            "context_ids": "context_ids",
        })
        self._auth = http_cfg.get("auth", None)
        self._logger = get_logger("pipeline.http_client")

    def query(self, question: str) -> RAGResponse:
        """Query the RAG system via HTTP and return a structured response.

        Args:
            question: The user question.

        Returns:
            RAGResponse with answer, retrieved contexts, and metadata.
        """
        url = f"{self._base_url}{self._query_endpoint}"
        body = self._build_request_body(question)

        start = time.perf_counter()
        try:
            self._logger.debug("HTTP query", extra={"url": url, "question": question[:80]})

            kwargs: Dict[str, Any] = {
                "headers": self._headers,
                "timeout": self._timeout,
            }
            if self._auth:
                kwargs["auth"] = self._resolve_auth(self._auth)

            if self._method == "POST":
                resp = requests.post(url, json=body, **kwargs)
            elif self._method == "GET":
                resp = requests.get(url, params=body, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {self._method}")

            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000

            self._logger.info(
                "HTTP query completed",
                extra={"latency_ms": round(latency_ms, 2), "status_code": resp.status_code},
            )

            return self._parse_response(data, latency_ms)

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self._logger.exception(
                "HTTP query failed",
                extra={"url": url, "question": question[:80], "latency_ms": round(latency_ms, 2)},
            )
            return RAGResponse(
                answer=f"[Error] {e}",
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    def _build_request_body(self, question: str) -> Dict[str, Any]:
        """Build the HTTP request body from the template.

        Args:
            question: The user question.

        Returns:
            Request body dict with {question} placeholders replaced.
        """
        body: Dict[str, Any] = {}
        for key, value in self._request_template.items():
            if isinstance(value, str):
                body[key] = value.replace("{question}", question)
            else:
                body[key] = value
        return body

    def _parse_response(self, data: Dict[str, Any], latency_ms: float) -> RAGResponse:
        """Parse the HTTP JSON response into a RAGResponse using the field mapping.

        Args:
            data: Raw JSON response from the RAG system.
            latency_ms: Request latency.

        Returns:
            Structured RAGResponse.
        """
        answer = self._extract_field(data, self._response_mapping.get("answer", "answer"))
        contexts = self._extract_list_field(
            data, self._response_mapping.get("contexts", "contexts")
        )
        context_ids = self._extract_list_field(
            data, self._response_mapping.get("context_ids", "context_ids")
        )

        return RAGResponse(
            answer=str(answer) if answer else "",
            retrieved_contexts=[str(c) for c in contexts],
            retrieved_ids=[str(i) for i in context_ids],
            latency_ms=latency_ms,
            metadata={"raw_response": data},
        )

    @staticmethod
    def _extract_field(data: Dict[str, Any], path: str) -> Any:
        """Extract a value from nested dict using dot-notation path.

        Args:
            data: Source dict.
            path: Dot-separated key path (e.g. "result.answer").

        Returns:
            The extracted value, or None if not found.
        """
        current: Any = data
        for key in path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    @staticmethod
    def _extract_list_field(data: Dict[str, Any], path: str) -> List[Any]:
        """Extract a list of values, supporting ``items[].field`` notation.

        Examples:
            - ``"contexts"`` → ``data["contexts"]``  (already a list)
            - ``"sources[].content"`` → ``[s["content"] for s in data["sources"]]``

        Args:
            data: Source dict.
            path: Dot-separated path, optionally with ``[]`` for list expansion.

        Returns:
            List of extracted values.
        """
        if "[]" in path:
            parts = path.split("[]")
            array_path = parts[0].rstrip(".")
            sub_path = parts[1].lstrip(".") if len(parts) > 1 else ""

            array_val = HttpRAGClient._extract_field(data, array_path) if array_path else data
            if not isinstance(array_val, list):
                return []

            if sub_path:
                return [
                    HttpRAGClient._extract_field(item, sub_path)
                    for item in array_val
                    if isinstance(item, dict)
                ]
            return list(array_val)

        val = HttpRAGClient._extract_field(data, path)
        if isinstance(val, list):
            return val
        return []

    @staticmethod
    def _resolve_auth(auth_config: Dict[str, str]) -> Any:
        """Resolve authentication from config.

        Args:
            auth_config: Auth config dict with 'type' and credentials.

        Returns:
            Auth object suitable for requests library.
        """
        auth_type = auth_config.get("type", "basic")
        if auth_type == "basic":
            return (auth_config.get("username", ""), auth_config.get("password", ""))
        elif auth_type == "bearer":
            return None  # handled via headers
        return None
