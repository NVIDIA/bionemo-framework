# Evo 2 Phage Design

This recipe fine-tunes Evo 2 on phage genomes, runs
[GDPO](https://arxiv.org/abs/2601.05242)—an NVIDIA-developed method for stable multi-reward policy
optimization—generates complete candidate genomes, and applies sequence-safety and design screens.
A key extension beyond the
[original published workflow](https://www.science.org/doi/10.1126/science.aec2657) is the
reinforcement learning (RL) stage, which optimizes generation toward user-defined computational
design criteria rather than relying only on post-generation filtering.

The included PhiX174 example follows Samuel King's recommended set of filters (1–6, 8, and 9) from
[Figure 2H](https://www.science.org/doi/10.1126/science.aec2657#F2). Their measurable constraints are
represented as separate graded or categorical RL objectives; reward full-credit regions, final
per-genome gates, and set-level diversity selection are distinct. See the
[PhiX174 GDPO score definitions](examples/README.md#current-phix174-gdpo-score-definitions) for the
objective definitions and hard-filter relationships.

The recipe includes agent skills led by the top-level `bionemo-phage-design` skill. They can adapt
the example launcher to the available GPU environment, help plan and implement related design
tasks, and monitor long-running jobs.

The current final screen combines circular-ORF AAI, core conservation, safety, and v3 syntenic
diversity. See the [final-filter contract](skills/bionemo-phage-design/references/arc-final-filter-contract.md)
for acceptance rules and the [screening evidence](docs/phix174-screening-evidence.md) for biological rationale.
The [ARC validation notes](docs/arc-validation.md) distinguish upstream-parity tests
from intentional scientific extensions and document remaining reproduction limits.

## Frozen rollout results

**Current state — 2026-10-22:** the fresh cosine-decay checkpoint **`fea26289`, step 900**
rollout and current v3 screening are complete. Seed 12 generated 10,000 genomes on one GB300
with batch size 64; all 10,000 received finite SFT likelihood scores.

| Cohort under the v3 screen                              | Raw generations | Passing observations | Final cluster representatives |
| ------------------------------------------------------- | --------------: | -------------------: | ----------------------------: |
| **Cosine step 900, fresh seed-12 rollout**              |      **10,000** |      **355 (3.55%)** |                        **16** |
| Earlier cosine pool, train 789–801 + validation 790/800 |          10,176 |          352 (3.46%) |                            16 |
| Canonical A/B step 400 (`1cfffb21`), rescored           |          10,000 |            6 (0.06%) |                             5 |

The current 355 passers comprise **351 original ARC-count passes and four structured K+X-only
passes**, with no overlap. Clustering at 99% nucleotide identity / 95% reciprocal coverage
leaves one 340-member cluster and 15 singletons; score summaries weight the 16 representatives
equally. The separate filter-7 diagnostic passes 27 designs and is not the headline gate.

Final yield is similar to the earlier cosine pool; these different checkpoint/cohort evaluations
are not a controlled comparison or evidence that the same clusters recur. Computational
acceptance does not establish experimental viability.
[Detailed results and artifact hashes](skills/bionemo-phage-design/references/case-study-results.md#2026-10-22-fresh-cosine-step-900-rollout-v3)
retain the denominators and provenance. The older step-1000 rollout is **held at 2,791/10,000
completed records**, not a finalized comparison cohort.

## Quick start

See the [example README](examples/README.md) for additional launch options and operational details.

The end-to-end run takes approximately 4 days on a server with 8 H100 GPUs, and uses 1.5TB of storage.

Run the following from `recipes/evo2_phage_gen` for the current `7b-base` end-to-end configuration.
All default prompts start at reference coordinate 1; see the
[prompt-origin rationale](examples/README.md#prompt-origin) for the related-assembly and SFT evidence.
The [frozen rollout results](#frozen-rollout-results) are separate checkpoint evaluations; the launcher defaults to 1,000 final designs.

```bash
./.ci_build.sh
./examples/phix174_8xh100.sh \
  --model-variant 7b-base  \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-8xh100-origin"
```

Like the Evo 2 recipe, `.ci_build.sh` creates an editable Python 3.12 virtual
environment with `--system-site-packages` and reuses the NVIDIA PyTorch
container's Torch, TorchVision, Triton, Transformer Engine, CUDA, and cuDNN
stack. The two recipes share the same Evo 2 runtime pins and native-extension
fallback; this recipe then adds NeMo-RL and phage workflow dependencies. The
example launcher sources `.ci_test_env.sh` itself; source it explicitly before
running individual recipe commands by hand.

Use tmux or a scheduler for long runs. `NUM_GPUS` defaults to 8, `NUM_CPUS` to `nproc`, and
`SFT_TENSOR_PARALLEL_SIZE` may adapt a measured smaller topology.
The [8×H100 example](examples/README.md) documents dry runs, an opt-in
[quick live E2E check](examples/README.md#quick-live-end-to-end-check), preparation-only mode, stage markers,
sampling overrides, outputs, and all RL objectives.

## Run with an agent

From the repository root, ask your preferred coding agent to adapt and run the workflow:

```text
Use $bionemo-phage-generation to adapt the PhiX174 example for this GB300 node, run and monitor it, and use the 7B-1M checkpoint instead of 7B-8K.
```

## References

- [PhiX174 phage-design publication](https://www.science.org/doi/10.1126/science.aec2657)
- [Evo 2 publication](https://doi.org/10.1038/s41586-026-10176-5)
- [Public Microviridae SFT checkpoint](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae)
- [Evo 2 model and checkpoint notes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/main/recipes/evo2_megatron/README.md)

## Acknowledgements

Thanks to Samuel King, Jessica Sacher, Jan Zheng, Avery Noonan, Michael Poon, and colleagues at
Tabula Bio, and to Eric Bastien and Nick Conley at Locus Biosciences, for discussions, guidance, and
feedback that shaped the recipe and its safety controls.

## Appendix: Filter rationale and evidence

The [screening evidence](docs/phix174-screening-evidence.md) preserves the selected-cohort
waterfall, length/coverage rationale, and experimental-outcome limitations.
The [accessory score definition](docs/accessory_genes.md) distinguishes the online K/X reward
from v3's independently measured final acceptance route; mutations within K do not supply X.
Historical screening versions remain in the
[case-study notes](skills/bionemo-phage-design/references/case-study-results.md).
