"""Retry structurally incomplete demos; preserve every attempt, never edit model text."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_showcase import convert

run = ROOT / "demo_runs"
deadline = time.monotonic() + 5400
while not (run / "summary.json").exists():
    if time.monotonic() > deadline:
        raise TimeoutError("Initial demo run did not finish before the retry deadline")
    time.sleep(15)
time.sleep(10)
rows = [json.loads(line) for line in (ROOT / "examples/research_questions.jsonl").read_text().splitlines() if line]
for seed in (29, 41):
    invalid = []
    for row in rows:
        path = run / "trajectories" / (row["id"] + ".json")
        try:
            convert(path, row["title"])
        except (ValueError, OSError) as exc:
            print(f"Retry needed: {row['id']}: {exc}", flush=True)
            invalid.append(row)
    if not invalid:
        print("All 20 demonstrations pass structural and citation-ID validation", flush=True)
        break
    attempt = run / f"retry_seed{seed}"
    attempt.mkdir(parents=True, exist_ok=True)
    queries = attempt / "queries.jsonl"
    queries.write_text("".join(json.dumps(row) + "\n" for row in invalid))
    subprocess.run([
        "bash", "scripts/run_demo.sh", "--input-path", str(queries), "--seed", str(seed),
        "--trajectory-dir", str(attempt / "trajectories"),
        "--output-jsonl", str(attempt / "reports.jsonl"), "--summary-json", str(attempt / "summary.json"),
    ], cwd=ROOT, check=True)
    for row in invalid:
        candidate = attempt / "trajectories" / (row["id"] + ".json")
        try:
            convert(candidate, row["title"])
        except (ValueError, OSError) as exc:
            print(f"Retry still invalid: {row['id']}: {exc}", flush=True)
            continue
        target = run / "trajectories" / candidate.name
        original = run / "original_attempts" / candidate.name
        original.parent.mkdir(exist_ok=True)
        if target.exists() and not original.exists():
            shutil.copy2(target, original)
        shutil.copy2(candidate, target)
        print(f"Accepted retry: {row['id']}, seed {seed}", flush=True)
else:
    print("Retry budget exhausted; inspect validation before publishing", flush=True)
