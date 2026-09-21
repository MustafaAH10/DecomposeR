from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # JSON strings may contain Unicode line-separator characters such as U+2028.
    # `splitlines()` treats those as row breaks and corrupts valid JSONL records.
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader if isinstance(row, dict)]


def extract_json_object(text: str) -> dict[str, Any] | list[Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    candidate_starts = [idx for idx, ch in enumerate(raw) if ch in "{["]
    for start in candidate_starts:
        try:
            payload, _ = decoder.raw_decode(raw[start:])
        except Exception:
            continue
        if isinstance(payload, (dict, list)):
            return payload
    return None


def train_val_split(
    rows: list[dict[str, Any]],
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    if val_ratio <= 0:
        return rows, []
    split_at = max(1, int(len(rows) * (1.0 - val_ratio)))
    split_at = min(split_at, len(rows))
    return rows[:split_at], rows[split_at:]
