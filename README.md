# DecomposeR

Planner-centric reinforcement learning for deep research with structure-aware
reward. Camera-ready release preparation for the accepted TMLR paper.

DecomposeR first plans a typed research graph, searches the web, revises the
graph once and chooses pages to fetch. It then writes evidence-grounded branch
reports in dependency order and synthesizes the final report. The released
checkpoint is a **merged Qwen3-8B model** used in both roles; the demo does not
load or switch between two separately released LoRA adapters.

## Start here

| Task | Location |
|---|---|
| Inspect saved benchmark reports and graphs without a GPU | `artifacts/evaluation/` |
| Reproduce the HealthBench subset without a model or API keys | Command below |
| Generate new research reports | `scripts/run_demo.sh` |
| Read the exact planner and answerer instructions | `decomposer_v4/prompts/` |
| Inspect training queries and RL rubrics | `artifacts/training_data/` |
| Review release gaps and provenance | `docs/RELEASE_AUDIT.md` |
| Read the original submitted documentation | `docs/SUBMITTED_README.md` |

## Layout

```text
decomposer_v4/         Submitted pipeline, prompts, configurations and evaluators
verl/                 Vendored training framework, with upstream license notices
artifacts/
  training_data/      2,325 SFT queries; 1,432 RL queries with rubrics
  evaluation/         100 DRBench, 500 HealthBench and 776 ResearchQA trajectories
examples/             New, explicitly separate demonstration questions
scripts/              Demo generation and release-validation utilities
docs/                 Release audit and archived submission documentation
```

The benchmark trajectories are lean: they preserve graphs, model outputs,
citations and timing, but omit full retrieved web-page content. These are
final-model artifacts, not a complete archive of all baselines, seeds or
ablations. DeepResearchBench includes aggregate RACE/FACT summaries rather
than per-item grader traces. Training queries are not full SFT completions.

## Public model checkpoint

[Merged DecomposeR Qwen3-8B checkpoint](https://huggingface.co/mustafaah/qwen3-8b-decomposer-v4-planner-answerer-rl-step358-merged)

For the new demonstrations, pin revision
`503e2bb12418b22259c0196700d3b0587ea00d42`. The checkpoint is public; a
Hugging Face token is not normally needed to download it.

Use a compatible CUDA/vLLM environment, then install the small inference overlay:

```bash
python3 -m pip install -r requirements-inference.txt
export JINA_API_KEY=...      # Search queries, up to five results per node
export SERPER_API_KEY=...    # Fetch selected URLs, at most four per revision
bash scripts/run_demo.sh
```

The complete demonstration run uses a 96 GB GPU. The script does not rent
hardware. Credentials stay in the process environment, never in Git or the
browser. OpenAI, Gemini, Exa, DeepSeek and W&B credentials are not required to
generate the demonstrations. Benchmark grading is a separate paid step.

## Reproduce the HealthBench subset

This preparation step needs only Python and PyYAML, not a GPU, model or API key.
Run from this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install PyYAML
python3 decomposer_v4/eval/prepare_healthbench_queries.py \
  --subset all --sample 500 --hard-fraction 0.2 --seed 17 \
  --output-path outputs/hb_queries.jsonl \
  --selected-healthbench-output-path outputs/hb_selected.jsonl \
  --summary-json outputs/hb_summary.json
```

The first invocation downloads the official dataset and may take several
minutes. To use previously downloaded files, supply `--input-path` and
`--hard-input-path`. Compare the selected IDs with
`artifacts/evaluation/healthbench/selected_prompt_ids.txt`.

## Scope and limitations

Graph validation, successful execution and well-formed citations do not prove
that every generated claim is correct. Inspect underlying sources before using
reports for consequential decisions. New showcase queries are demonstrations,
not additional benchmark measurements.

The top-level `requirements.txt` retains the submitted training dependencies.
Do not install that full stack merely to inspect saved results or prepare a
subset. Full training also requires teacher completions/prepared evidence not
included in this query-only data release.

## Licensing and attribution

This repository remains private while release preparation is underway. Preserve
the licenses in `verl/` and `decomposer_v4/eval/simple-evals/`. A license for
original DecomposeR code and final paper citation are still to be supplied by
the authors. Benchmark data and cited web content retain their upstream terms.
