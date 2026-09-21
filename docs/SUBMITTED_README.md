# DecomposeR: Software, Training Data, and Evaluation Artifacts

This bundle contains everything needed to understand, reproduce, and inspect the
DecomposeR pipeline and its reported results:

1. **Code** for the full pipeline (cold-start SFT, planner GRPO, planner-graph
   regeneration, answerer preparation, answerer GRPO) and the three benchmark
   evaluation runners (DeepResearchBench, ResearchQA-Mini, HealthBench).
2. **Training data**: the RL and SFT queries (the RL queries include their
   query-specific rubrics).
3. **Evaluation artifacts** for all three benchmarks: the exact queries used, the
   model's cited reports, per-query trajectories, and the official judged outputs.

---

## Repository layout

```
README.md                     <- this file
requirements.txt
decomposer_v4/                <- pipeline + evaluation code
  configs/                    paper-matched SFT and GRPO configs
  prompts/                    planner, answerer, rubric, and judge prompts
  train/                      SFT, adapter merge, GRPO launch, VERL loop
  reward/                     planner and answerer reward implementations
  schemas/                    typed-DAG / trajectory validation
  data/                       builders for SFT/RL artifacts + generation code
  retrieval/                  Jina Search and Serper fetch clients
  eval/                       benchmark preparation, generation, grading
verl/                         vendored VERL package (no git history)

artifacts/
  training_data/                              training queries only (see below)
    rl_queries_1432.jsonl                     1,432 RL queries with query-specific rubrics
    sft_queries_2325.csv                      2,325 SFT queries
  evaluation/
    healthbench/              queries, reports, graded outputs, score, selected IDs, trajectories_lean/
    deepresearchbench/        queries (en+zh), RACE/FACT results, trajectories_lean/{en,zh}
    researchqa_mini/          queries, cited responses, coverage scores, trajectories_lean/
```

### Training data (queries only)

The 1,432 RL queries (each with its
query-specific rubrics) and the 2,325 SFT queries are shared. The source breakdown matches Table 5
of the paper:

| Source | Pool | SFT used | RL used |
|---|---|---|---|
| OpenScholar | 2,577 | 1,554 | 897 |
| SearchArena | 1,039 | 558 | 403 |
| ScholarQA | 384 | 213 | 132 |
| **Total** | **4,000** | **2,325** | **1,432** |

### Lean evaluation trajectories

Per-query trajectories (`trajectories_lean/`) retain the model-generated objects and
metadata (initial and revised typed-DAG graphs, planner outputs, answerer node/branch
reports, final answer, citation catalog, fetch requests, timing) but omit the raw
fetched-page and search-passage text, again to avoid redistributing copyrighted web
content.

---

## Reported numbers (this bundle)

Every headline number in the paper is reproducible from the included judged outputs:

| Benchmark | Metric | Value | Source file |
|---|---|---|---|
| HealthBench | rubric score | **42.0** (0.4198, bootstrap std 1.6) | `artifacts/evaluation/healthbench/healthbench_score.json` |
| DeepResearchBench | RACE (English) | **42.9** | `artifacts/evaluation/deepresearchbench/race_en.txt` |
| DeepResearchBench | RACE (Chinese) | **40.7** | `artifacts/evaluation/deepresearchbench/race_zh.txt` |
| DeepResearchBench | FACT citation acc. (En/Zh) | **57.8% / 73.4%** | `artifacts/evaluation/deepresearchbench/fact_{en,zh}.txt` |
| ResearchQA-Mini | rubric coverage | **71.4** (0.7143) | `artifacts/evaluation/researchqa_mini/coverage_scores.json` |

The combined DeepResearchBench RACE score reported in the main table (41.8) is the mean
of the English and Chinese scores. A train/eval decontamination analysis (reported in the
paper's appendix) finds **0 exact matches, 0 shared 13-grams, and no near-duplicates**
(max embedding cosine 0.73–0.75) between the 3,633 unique training queries and every
evaluation set.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r decomposer_v4/eval/requirements_eval.txt
python3 -m pip install -e ./verl

export JINA_API_KEY=...      # search
export SERPER_API_KEY=...    # URL fetch
export OPENAI_API_KEY=...    # judges / rubric generation
```

## Model weights

The base model (Qwen3-8B) is public. The merged DecomposeR-8B planner+answerer-RL
checkpoint is released **anonymously** to preserve double-blind review, and is hosted
separately from this code bundle because of its size:

> **Anonymized weights:** https://anonymous-hf.com/a/j1xjye6x1aki/

All three evaluation runners read the model from the `MODEL` environment variable and
default to the local path `checkpoints/decomposer_qwen3_8b_final_merged`. Download the
checkpoint into that path and the runners work with no further configuration:

```bash
wget "https://anonymous-hf.com/api/a/j1xjye6x1aki/download/" -O repo.zip
unzip repo.zip -d checkpoints/decomposer_qwen3_8b_final_merged
# if the archive extracts into a nested subfolder, either move the model files up so that
# config.json sits directly under checkpoints/decomposer_qwen3_8b_final_merged, or set:
#   export MODEL="$(pwd)/checkpoints/decomposer_qwen3_8b_final_merged/<subfolder>"
```

A quick sanity check that the weights load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "checkpoints/decomposer_qwen3_8b_final_merged"
model = AutoModelForCausalLM.from_pretrained(p)
tok = AutoTokenizer.from_pretrained(p)
```

## Reproducing the HealthBench subset (no model or API keys)

Creating the HealthBench subset is fully decoupled from generation and grading: it needs
**only Python 3 and PyYAML** (`pip install pyyaml`) and no model weights or OpenAI / Jina
/ Serper keys. First download the two official HealthBench files (~68 MB; `curl` shows a
progress bar), then select the exact 500-example subset used in the paper (20% drawn from
the official hard split, sampling seed 17):

```bash
# 1) download the official HealthBench source files (~68 MB total)
curl -L -o healthbench_all.jsonl  "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
curl -L -o healthbench_hard.jsonl "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"

# 2) select the subset (instant)
python3 decomposer_v4/eval/prepare_healthbench_queries.py \
  --subset all --sample 500 --hard-fraction 0.2 --seed 17 \
  --input-path healthbench_all.jsonl --hard-input-path healthbench_hard.jsonl \
  --output-path hb_queries.jsonl \
  --selected-healthbench-output-path hb_selected.jsonl \
  --summary-json hb_summary.json
```

> If you omit `--input-path` / `--hard-input-path`, the script downloads these files
> itself, but silently (no progress bar), so it can look frozen for a few minutes on the
> first run. Downloading them up front with `curl` (step 1) avoids that.

Verify it matches the subset used in the paper (the 500 ids in
`artifacts/evaluation/healthbench/selected_prompt_ids.txt`):

```bash
python3 - <<'PY'
import json
new = {json.loads(l)["prompt_id"] for l in open("hb_selected.jsonl")}
ref = {l.strip() for l in open("artifacts/evaluation/healthbench/selected_prompt_ids.txt") if l.strip()}
print("identical:", new == ref, "| sizes:", len(new), len(ref))
PY
```

This prints `identical: True`. The other benchmarks (DeepResearchBench, ResearchQA-Mini) use
the full set, so no subset step is needed.

## Running the full end-to-end evaluation (optional)

Generating and grading reports (as opposed to just forming the subset above) additionally
requires the **Setup** above, the **model weights**, and the API keys. Each runner
defaults `MODEL` to `checkpoints/decomposer_qwen3_8b_final_merged`:

```bash
bash decomposer_v4/eval/run_healthbench_eval.sh                # forms the subset, generates, grades
bash decomposer_v4/eval/run_deep_research_bench_full_eval.sh   # needs a deep_research_bench/ checkout
bash decomposer_v4/eval/ResearchQA/run_researchqa_mini_final_8b.sh
```

DeepResearchBench expects an external checkout at `deep_research_bench/`; ResearchQA-Mini
downloads through the Hugging Face datasets interface; HealthBench downloads the official
simple-evals JSONL and grades with the included `decomposer_v4/eval/simple-evals/`
implementation. Judge model/version strings used in the paper: RACE `gpt-5.5`, FACT
`gpt-5.4-mini`, ResearchQA-Mini `gpt-4.1-mini`, HealthBench `gpt-4.1-2025-04-14`.

