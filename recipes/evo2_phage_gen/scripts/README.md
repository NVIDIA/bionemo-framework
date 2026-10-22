# Maintained workflow scripts

The default final rollout uses **v3 syntenic diversity**: original ARC count diversification **or a separately evidenced, structured K+X “add one gene” category**, with circular-ORF AAI, nine-core completeness/order/copies and exact safety PASS required for both. Counts such as 11/13 alone never pass. Original-inventory AAI remains diagnostic and architecture diversification stays off. Retain v2 for original ARC-count-only screening and v1 for both AAI gates; report these contracts separately.

The final-rollout acceptance contract is [ARC filters with circular-AAI correction, architecture off, plus safety and nine-core conservation](../skills/bionemo-phage-design/references/arc-final-filter-contract.md). Historical adapted-core results do not measure the original ARC synteny/count endpoint.

Most reusable behavior lives in the `bionemo.evo2_phage_gen` package and its installed command-line
entry points. This directory retains two shell helpers for the parallel, resumable calibration grid:

- `calibration/run_sft_sampling_sweep.sh` generates the temperature and prefix-length cells for a
  selected checkpoint. `STOP_ON_EOS=1` permits sampled EOD before `TARGET_LENGTH`, a total DNA cap
  including prompt bases; `0` retains fixed-length generation. The termination mode is saved with
  the sweep configuration and cannot change on resume. Longer circular prompts use the existing
  named anchors and total prompt lengths, with correspondingly shorter generation budgets.
- `calibration/run_sampling_calibration_scoring.sh` scores those cells with the external-QC and
  sequence-safety objectives, checks similarity to the reference and training corpus, and writes
  the selection evidence. Its safety manifest, policy, host domain, and confirmed host evidence
  are required and must match online RL; unexplained unavailable measurements fail closed.

Calibration summaries report reward components, measurement support, nucleotide/safety gates,
and within-setting diversity. Selection evidence includes uncertainty on aggregate reward and
target signal (the mean of tropism and required-gene rewards). Full-QC yield requires the final-screening artifacts; it is not inferred from
calibration reward columns.

The [8×H100 PhiX174 example](../examples/README.md) invokes both helpers as part of the complete
agent-free workflow. Their shell tests mirror the directory layout under `tests/scripts/`.
