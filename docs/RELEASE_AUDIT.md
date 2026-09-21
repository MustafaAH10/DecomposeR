# Supplementary-material audit

Source: `10356_DecomposeR_Planner_Centr_Supplementary Material.zip`.
The source archive is preserved unchanged outside this checkout. This repository
starts from its contents, not the development repository's working tree.

## Included

- Core `decomposer_v4` inference, retrieval, prompts, graph validation, reward,
  SFT and RL code, with vendored VERL and its license notices.
- Training queries: 2,325 SFT rows and 1,432 RL rows (including RL rubrics).
  These are not full teacher trajectories or ready-to-train SFT completions.
- Final-model evaluation trajectories: 100 DeepResearchBench, 500 HealthBench,
  and 776 ResearchQA-Mini. Lean trajectories omit full retrieved web pages.
- HealthBench per-item grades, ResearchQA coverage outputs, and aggregate
  English/Chinese RACE and FACT summaries. DeepResearchBench per-item judge
  records are not in this archive.

## Missing or recommended before a public camera-ready release

1. Choose a license for original DecomposeR code. Upstream VERL/simple-evals
   licenses are present, but do not license the original project automatically.
2. Add a model card to the public Hugging Face checkpoint: training stages,
   intended use, limitations, prompting requirements, and upstream license.
3. Add the final paper citation and author list after the final bibliographic
   information is settled; do not guess a DOI or publication date.
4. Preserve environment versions for the successful demo run. The submitted
   requirements are largely unpinned and do not establish a reproducible GPU
   environment on their own.
5. If releasing statistical/robustness studies, include the actual baseline,
   repeated-run, alternate-judge and human annotation records. This ZIP alone
   does not substantiate all such studies.
6. Add per-item DeepResearchBench RACE/FACT outputs from the development
   outputs directory if redistribution is permitted.

## Comparison with the development `decomposer_v4` directory

The ZIP already includes the canonical end-to-end report runner and the core
model backends, so the local HF/vLLM single-trajectory wrappers are conveniences,
not essential missing inference implementations. Development-only runbooks,
compact/small-GPU training configs, token/fetch diagnostics and reward smoke tests
would improve a later training release. Do not copy them blindly: some embed
development paths and depend on unpublished prepared training artifacts.

Local planner prompts and several runners differ from the submitted versions.
This release retains the submitted prompts and fixes instead of silently
replacing them with the current development files. Demo tooling and automated
artifact checks are added separately under `scripts/` and `examples/`.

The independent `deep_research_bench/` evaluator is not bundled. It must be
installed separately to rerun RACE/FACT. No model weights, API credentials,
training logs, caches or nested Git histories are added to this repository.
