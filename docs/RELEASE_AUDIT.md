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

## License and citation

Original DecomposeR code is licensed under [Apache-2.0](../LICENSE).
Third-party license notices are preserved. Paper metadata and the recommended
citation are available in [`CITATION.cff`](../CITATION.cff) and the
[README](../README.md#citation).

## Artifact coverage and reproducibility

- The submitted requirements are largely unpinned and do not establish a
  reproducible GPU environment on their own. Record environment versions when
  reproducing training or inference.
- The archive contains final-model artifacts, rather than the full baseline,
  repeated-run, alternate-judge, or human annotation records.
- Per-item DeepResearchBench RACE/FACT judge outputs are not included; the
  release contains aggregate summaries.
- Checkpoint usage and limitations are described in the README. Consult the
  public checkpoint's model card for any additional model-specific information.

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
artifact checks are added separately under `demo/scripts/` and `demo/examples/`.

The independent `deep_research_bench/` evaluator is not bundled. It must be
installed separately to rerun RACE/FACT. No model weights, API credentials,
training logs, caches or nested Git histories are added to this repository.
