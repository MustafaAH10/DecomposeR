from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rubric:
    title: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class SearchPassage:
    passage_id: str
    node_id: str
    title: str
    url: str
    text: str
    score: float = 0.0
    source: str = ""
    published_date: str = ""
    author: str = ""
    language: str = ""
    language_score: float = 0.0
    token_count: int = 0


@dataclass(frozen=True)
class SearchExecution:
    node_id: str
    query: str
    passages: tuple[SearchPassage, ...]


@dataclass(frozen=True)
class FetchRequest:
    source_node_id: str
    url: str
    reason: str = ""


@dataclass(frozen=True)
class SpanRecord:
    span_id: str
    role: str
    turn_type: str
    messages: tuple[dict[str, str], ...]
    completion: str
    metadata: dict[str, str] = field(default_factory=dict)
