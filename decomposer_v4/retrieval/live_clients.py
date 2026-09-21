from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests


JINA_SEARCH_URL = "https://s.jina.ai/"
SERPER_FETCH_URLS = (
    "https://scrape.serper.dev",
    "https://google.serper.dev/scrape",
)


def _load_dotenv_if_available() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    text: str
    source: str = "web"
    rank: int = 0
    score: float = 0.0
    published_date: str = ""
    author: str = ""
    highlights: tuple[str, ...] = field(default_factory=tuple)
    language: str = ""
    language_score: float = 0.0
    token_count: int = 0


class LiveSearchClient(Protocol):
    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        raise NotImplementedError

    def fetch(self, url: str, max_characters: int = 4000) -> SearchResult | None:
        raise NotImplementedError


def _safe_text(*values: object) -> str:
    for value in values:
        if isinstance(value, list):
            joined = "\n".join(" ".join(str(item or "").split()).strip() for item in value if str(item or "").strip())
            if joined.strip():
                return joined.strip()
            continue
        text = " ".join(str(value or "").split()).strip()
        if text:
            return text
    return ""


def _truncate(text: str, limit: int) -> str:
    if int(limit or 0) <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _normalize_search_query(query: str) -> str:
    normalized = " ".join(str(query or "").split()).strip()
    if not normalized:
        return normalized
    # Some providers reject aggressively quoted queries; soften them before a retry.
    normalized = normalized.replace('"', "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {url}, got: {type(parsed)}")
    return parsed


def _parse_jina_blocks(text: str) -> list[dict[str, str]]:
    blocks: dict[int, dict[str, str]] = {}
    pattern = re.compile(r"^\[(\d+)\]\s+([^:]+):\s*(.*)$")
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue
        index = int(match.group(1))
        key = match.group(2).strip().lower()
        value = match.group(3).strip()
        blocks.setdefault(index, {})[key] = value
    return [blocks[index] for index in sorted(blocks)]


@dataclass
class JinaSearchClient:
    api_key: str
    timeout: float = 30.0
    respond_with: str = "no-content"

    @classmethod
    def from_env(cls) -> "JinaSearchClient":
        _load_dotenv_if_available()
        api_key = os.environ.get("JINA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("JINA_API_KEY is not set.")
        return cls(api_key=api_key)

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Respond-With": self.respond_with,
        }
        blocks: list[dict[str, str]] = []
        attempts = [str(query or "").strip()]
        normalized_query = _normalize_search_query(query)
        if normalized_query and normalized_query not in attempts:
            attempts.append(normalized_query)

        for attempt in attempts:
            if not attempt:
                continue
            try:
                response = requests.get(
                    JINA_SEARCH_URL,
                    params={"q": attempt},
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                blocks = _parse_jina_blocks(response.text)
                break
            except requests.exceptions.HTTPError as exc:
                response = exc.response
                # Treat provider-side query rejections as empty results so training can continue.
                if response is not None and response.status_code in {400, 401, 403, 422, 429, 500, 502, 503, 504}:
                    continue
                return []
            except requests.exceptions.RequestException:
                return []

        results: list[SearchResult] = []
        for idx, block in enumerate(blocks[: max(int(num_results), 1)], start=1):
            title = _safe_text(block.get("title"))
            url = _safe_text(block.get("url source"), block.get("url"))
            description = _safe_text(block.get("description"), title)
            if not title or not url or not description:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    text=description,
                    source="jina_search",
                    rank=idx,
                    published_date=_safe_text(block.get("date")),
                )
            )
        return results

    def fetch(self, url: str, max_characters: int = 4000) -> SearchResult | None:
        del url, max_characters
        return None


@dataclass
class SerperFetchClient:
    api_key: str
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "SerperFetchClient":
        _load_dotenv_if_available()
        api_key = os.environ.get("SERPER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("SERPER_API_KEY is not set.")
        return cls(api_key=api_key)

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        del query, num_results
        return []

    def fetch(self, url: str, max_characters: int = 4000) -> SearchResult | None:
        payload = {"url": url}
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        last_error: Exception | None = None
        for endpoint in SERPER_FETCH_URLS:
            try:
                parsed = _post_json(endpoint, headers=headers, payload=payload, timeout=self.timeout)
            except Exception as exc:
                last_error = exc
                continue
            text = _safe_text(
                parsed.get("text"),
                parsed.get("markdown"),
                parsed.get("content"),
                parsed.get("extract"),
            )
            if not text:
                continue
            metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
            return SearchResult(
                title=_safe_text(metadata.get("title"), parsed.get("title"), url),
                url=url,
                text=_truncate(text, max_characters),
                source="serper_fetch",
                published_date=_safe_text(metadata.get("publishedDate"), parsed.get("publishedDate")),
                author=_safe_text(metadata.get("author"), parsed.get("author")),
                language=_safe_text(metadata.get("language")),
                language_score=_safe_float(metadata.get("languageScore")),
                token_count=_safe_int(metadata.get("tokenCount")),
            )
        if last_error is not None:
            return None
        return None


@dataclass
class MultiSearchClient:
    clients: tuple[LiveSearchClient, ...]
    fetch_clients: tuple[LiveSearchClient, ...] | None = None

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        deduped: dict[str, SearchResult] = {}
        for client in self.clients:
            try:
                client_results = client.search(query, num_results=num_results)
            except Exception:
                client_results = []
            for result in client_results:
                current = deduped.get(result.url)
                if current is None or len(result.text) > len(current.text):
                    deduped[result.url] = result
        ordered = sorted(deduped.values(), key=lambda item: (item.rank, -item.score, item.url))
        return ordered[:num_results]

    def fetch(self, url: str, max_characters: int = 4000) -> SearchResult | None:
        for client in (self.fetch_clients or self.clients):
            try:
                result = client.fetch(url, max_characters=max_characters)
            except Exception:
                result = None
            if result is not None and result.text.strip():
                return result
        return None
