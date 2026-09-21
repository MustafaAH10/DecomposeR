#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from decomposer_v4.schemas.trajectory import Rubric
from decomposer_v4.utils.io import ROOT


def resolve_path(raw_path: str | Path | None) -> str | None:
    if raw_path is None:
        return None
    raw = str(raw_path).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return str(path.resolve())
    return str((ROOT / path).resolve())


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def serialize_rubrics(rubrics: Sequence[Rubric] | Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for rubric in rubrics or ():
        if isinstance(rubric, Rubric):
            serialized.append(
                {
                    "title": rubric.title,
                    "description": rubric.description,
                    "weight": float(rubric.weight),
                }
            )
            continue
        if isinstance(rubric, dict):
            serialized.append(
                {
                    "title": str(rubric.get("title", "")),
                    "description": str(rubric.get("description", "")),
                    "weight": float(rubric.get("weight", 1.0) or 1.0),
                }
            )
    return serialized


def serialize_json_field(value: Any, default: Any) -> str:
    payload = value if value is not None else default
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(default, ensure_ascii=False, sort_keys=True)


def deserialize_rubrics(raw_rubrics: Sequence[dict[str, Any]] | str | None) -> tuple[Rubric, ...]:
    payload: Sequence[dict[str, Any]] | None
    if isinstance(raw_rubrics, str):
        try:
            loaded = json.loads(raw_rubrics)
        except Exception:
            loaded = []
        payload = loaded if isinstance(loaded, list) else []
    elif isinstance(raw_rubrics, Sequence):
        payload = raw_rubrics  # type: ignore[assignment]
    else:
        payload = []

    parsed: list[Rubric] = []
    for rubric in payload or ():
        if not isinstance(rubric, dict):
            continue
        parsed.append(
            Rubric(
                title=str(rubric.get("title", "")),
                description=str(rubric.get("description", "")),
                weight=float(rubric.get("weight", 1.0) or 1.0),
            )
        )
    return tuple(parsed)


def make_raw_prompt(query: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": str(query)}]


def write_parquet_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Writing VERL datasets requires pyarrow.") from exc

    row_list = list(rows)
    if not row_list:
        raise ValueError("Refusing to write an empty parquet dataset.")

    output_path = ensure_parent_dir(path)
    table = pa.Table.from_pylist(row_list)
    pq.write_table(table, output_path)
    return output_path


def latest_global_step_dir(root: str | Path) -> Path | None:
    resolved = Path(root).resolve()
    if not resolved.exists():
        return None
    candidates = sorted(
        (
            path
            for path in resolved.glob("global_step_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        ),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not candidates:
        return None
    return candidates[-1]
