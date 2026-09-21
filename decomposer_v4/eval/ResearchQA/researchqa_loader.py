from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.utils.io import write_json


@dataclass
class CitationMetadata:
    first_author: str = ""
    title: str = ""
    url: str = ""
    year: int | None = None

@dataclass
class RubricItem:
    rubric_item: str
    type: list[str]
    citation_metadata: Optional[CitationMetadata] = None

@dataclass
class ResearchQAItem:
    id: str
    general_domain: str
    subdomain: str
    field: str
    query: str
    date: str
    rubric: List[RubricItem]


def _normalize_date(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    return str(value)


def _normalize_citation_metadata(value: Any) -> CitationMetadata | None:
    if not isinstance(value, dict):
        return None
    year = value.get("year")
    normalized_year: int | None = None
    if year is not None and str(year).strip():
        try:
            normalized_year = int(year)
        except Exception:
            normalized_year = None
    return CitationMetadata(
        first_author=str(value.get("first_author", "")).strip(),
        title=str(value.get("title", "")).strip(),
        url=str(value.get("url", "")).strip(),
        year=normalized_year,
    )


def _normalize_rubric_item(value: Any) -> RubricItem:
    payload = value if isinstance(value, dict) else {}
    rubric_types = payload.get("type", [])
    if isinstance(rubric_types, str):
        normalized_types = [rubric_types] if rubric_types.strip() else []
    elif isinstance(rubric_types, list):
        normalized_types = [str(piece).strip() for piece in rubric_types if str(piece).strip()]
    else:
        normalized_types = []
    return RubricItem(
        rubric_item=str(payload.get("rubric_item", "")).strip(),
        type=normalized_types,
        citation_metadata=_normalize_citation_metadata(payload.get("citation_metadata")),
    )


def _normalize_item(item: Any) -> ResearchQAItem:
    payload = item if isinstance(item, dict) else {}
    rubric_rows = payload.get("rubric", [])
    rubric = [_normalize_rubric_item(row) for row in rubric_rows if isinstance(row, dict)]
    return ResearchQAItem(
        id=str(payload.get("id", "")).strip(),
        general_domain=str(payload.get("general_domain", "")).strip(),
        subdomain=str(payload.get("subdomain", "")).strip(),
        field=str(payload.get("field", "")).strip(),
        query=str(payload.get("query", "")).strip(),
        date=_normalize_date(payload.get("date", "")),
        rubric=rubric,
    )


def researchqa_item_to_dict(item: ResearchQAItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "general_domain": item.general_domain,
        "subdomain": item.subdomain,
        "field": item.field,
        "query": item.query,
        "date": item.date,
        "rubric": [
            {
                "rubric_item": rubric_item.rubric_item,
                "type": list(rubric_item.type),
                "citation_metadata": None
                if rubric_item.citation_metadata is None
                else {
                    "first_author": rubric_item.citation_metadata.first_author,
                    "title": rubric_item.citation_metadata.title,
                    "url": rubric_item.citation_metadata.url,
                    "year": rubric_item.citation_metadata.year,
                },
            }
            for rubric_item in item.rubric
        ],
    }

def load_researchqa_data(json_path: str) -> List[ResearchQAItem]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [_normalize_item(item) for item in data if isinstance(item, dict)]


def load_researchqa_split(split: str = "test_mini", cache_dir: str | None = None) -> List[ResearchQAItem]:
    from datasets import load_dataset

    dataset = load_dataset("realliyifei/ResearchQA", split=split, cache_dir=cache_dir)
    return [_normalize_item(item) for item in dataset]


def export_researchqa_split(split: str, output_path: str | Path, cache_dir: str | None = None) -> Path:
    items = load_researchqa_split(split=split, cache_dir=cache_dir)
    target = Path(output_path)
    write_json(target, [researchqa_item_to_dict(item) for item in items])
    return target
