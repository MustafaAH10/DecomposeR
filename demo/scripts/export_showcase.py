"""Export explicit model fields only; never expose prompts, credentials or local paths."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from decomposer_v4.schemas.graph import validate_graph


def convert(path, title=None, historical=False):
    d = json.loads(path.read_text())
    graph = d.get("final_graph") or d.get("parsed_graph")
    valid, details = validate_graph(graph)
    if not valid or d.get("error"):
        raise ValueError(f"Invalid trajectory {path.name}: {details}")
    initial_valid, initial_details = validate_graph(d.get("initial_graph"))
    if not initial_valid:
        raise ValueError(f"Invalid initial graph {path.name}: {initial_details}")
    outputs = d.get("node_outputs", {})
    missing = [n["id"] for n in graph["nodes"] if n["type"] != "search" and not outputs.get(n["id"], "").strip()]
    if missing or not d.get("final_answer", "").strip():
        raise ValueError(f"Incomplete outputs {path.name}: {missing}")
    catalog = d.get("citation_catalog", {})
    used = set(re.findall(r"\bN[A-Za-z0-9_]+-[RF][0-9]+\b", "\n".join(outputs.values())))
    unknown = sorted(used - set(catalog))
    if unknown:
        raise ValueError(f"Unknown citations {path.name}: {unknown}")
    citations = {k: {f: c.get(f, "") for f in ("title", "url", "kind", "source_node_id")} for k, c in catalog.items()}
    for c in citations.values():
        if not c["url"].startswith(("https://", "http://")):
            c["url"] = ""
    return {"id": "historical-example" if historical else d["query_id"],
            "title": title or "Trees as a nitrogen safety net", "query": d["query"],
            "provenance": "Saved paper example · not a new run" if historical else "New checkpoint demonstration",
            "initial_graph": d["initial_graph"], "graph": graph,
            "outputs": outputs, "citations": citations,
            "evidence": {k: v for k, v in d.get("search_node_outputs", {}).items() if isinstance(v, str)},
            "execution_waves": [v["node_ids"] for v in d.get("timings", {}).get("answer_execution_turns", {}).values()],
            "retrieval_status": {"requested": len(d.get("fetch_requests", [])),
                                 "succeeded": sum(bool(v.get("succeeded")) for v in d.get("timings", {}).get("fetch_timings", {}).values())},
            "answer": d["final_answer"], "fetches": d.get("fetch_requests", []),
            "timings": {k: v for k, v in d.get("timings", {}).items() if isinstance(v, (int, float))},
            "validation": {"valid_graph": True, "complete_nodes": True, "known_citations": True,
                           "claim_accuracy": "Not guaranteed; consult linked sources"}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--historical", action="store_true")
    p.add_argument("--skip-invalid", action="store_true", help="Preview only: print rejected IDs and export validated runs")
    a = p.parse_args()
    rows = [json.loads(s) for s in (ROOT / "demo/examples/research_questions.jsonl").read_text().split("\n") if s]
    titles = {r["id"]: r["title"] for r in rows}
    files = [a.source] if a.source.is_file() else sorted(a.source.glob("*.json"))
    data = []
    for f in files:
        try:
            data.append(convert(f, titles.get(f.stem), a.historical))
        except ValueError as exc:
            if not a.skip_invalid:
                raise
            print(f"REJECTED {f.stem}: {exc}")
    a.destination.parent.mkdir(parents=True, exist_ok=True)
    a.destination.write_text(json.dumps(data, ensure_ascii=False))
    print(f"Exported {len(data)} validated trajectories")
