from __future__ import annotations

from dataclasses import dataclass, field

from decomposer_v4.schemas.trajectory import Rubric


@dataclass(frozen=True)
class SFTConversationRecord:
    record_id: str
    query: str
    stage: str
    messages: tuple[dict[str, str], ...]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RLQueryRecord:
    record_id: str
    query: str
    rubrics: tuple[Rubric, ...]
    task_type: str = "deep_research"
    metadata: dict[str, str] = field(default_factory=dict)
