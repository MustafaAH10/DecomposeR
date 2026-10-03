"""Offline integrity checks; no model, credentials or network required."""
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "decomposer_v4").rglob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
expected = {"deepresearchbench": 100, "healthbench": 500, "researchqa_mini": 776}
for name, count in expected.items():
    folder = ROOT / "artifacts" / "evaluation" / name
    trajectories = list((folder / "trajectories_lean").rglob("*.json"))
    assert len(trajectories) == count, (name, len(trajectories))
    for path in trajectories:
        data = json.loads(path.read_text())
        assert isinstance(data, dict), path
    queries = [json.loads(line) for line in (folder / "queries.jsonl").read_text().split("\n") if line.strip()]
    assert len(queries) == count, (name, "queries", len(queries))
    print(f"{name}: {count} queries and {count} parseable saved trajectories")
data = ROOT / "artifacts" / "training_data"
assert sum(bool(line.strip()) for line in (data / "rl_queries_1432.jsonl").read_text().split("\n")) == 1432
with (data / "sft_queries_2325.csv").open(newline="") as stream:
    assert sum(1 for _ in csv.DictReader(stream)) == 2325
print("Training query counts and Python syntax: OK")
