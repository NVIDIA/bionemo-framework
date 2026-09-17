---
name: bionemo-phage-design-calibrate-rl-sampling
description: Use after selecting an Evo 2 phage SFT checkpoint and defining RL objectives to calibrate prompt format, sampling settings, and train/validation prompt mixtures.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Calibrate Phage RL Sampling

Use the selected SFT checkpoint's actual serialization and the project's scientific endpoint. For a faithful rerun, carry the user's reviewed sampling selection forward; fresh calibration does not override it. For a new selection, compare paired-seed samples across plausible settings and choose a robust quality/diversity region rather than the noisiest maximum.

The [PhiX example README](../../examples/README.md) describes commands, the review stop, selection schema, and resume markers. An explicit `--sampling-selection` is a reviewed override, not the outcome of fresh calibration. If selection is delegated, inspect `calibration/scoring/selection-evidence.csv` and neighboring score/novelty evidence, write `calibration/sampling-selection.yaml`, and continue. No further acknowledgment is needed within the delegated scope.

For a reduced execution check, the example's `CALIBRATION_PROMPTS` sets both
generated and expected scored rows per cell; `--quick-e2e` uses eight. When calling
the generation and scoring scripts separately, keep `NUM_PROMPTS` and
`EXPECTED_RECORDS` equal rather than inheriting the scorer's 64-row default.

## Construct prompts

Reconstruct conditioning, orientation, wrappers, tokenization, BOS/EOS, padding, and the continuation boundary from SFT. Use only cues the model saw.

Prompt bases are fixed, not designed. Use the objective plan's exclusions, including circular intervals that wrap the origin. Start with the shortest workable prompt; report its bases and genome fraction rather than scaling length by genome size. Longer prompts need evidence that they help. If a preserved required ORF is never reached from neutral anchors, a minimal start-seeded stratum can be tested alongside an unseeded stratum; record what it fixes.

For confirmed circular or headful/pac genomes, first inspect the collection's [start-site variability evidence](../bionemo-phage-design-collect-genomes/references/collection-guidance.md#start-site-variability): aligned starts/strands among relatives, submission or annotation conventions, and actual SFT start exposure or rotation augmentation. Biological permission to rotate does not imply that arbitrary starts generalize. If the corpus favors a canonical start, use that context; evaluate alternate rotations in a paired calibration before deploying them. Inspect overlap-aware CDS occupancy and annotated regulatory features at any candidate anchor. Summed CDS occupancy can exceed 100% for overlapping genes; missing annotations do not establish neutrality. For fixed-end dsDNA, reconstruct the biological termini and packaged/deposited form first, then compare the forward and reverse-complement terminal prompts rather than internal rotations.

The default PhiX calibration, RL, validation, and final rollout use coordinate 1 only, with 16/24-base prompts. Related SFT assemblies favor that deposited start and the released preprocessing adds no rotations; see the [counts and evidence limits](../../examples/README.md#prompt-origin). Coordinate 1 is not a claim about the biological replication origin. Keep circular ORF handling and rotation-invariant biological scores independently of this prompt choice.

For a requested fixed scaffold on a circular genome, the existing named-anchor prompt builder can supply an interval that wraps coordinate one. `prompt_lengths` counts the entire fixed interval, including the reference tail. Subtract additional prompt bases from the generation budget when preserving a total DNA cap. Check that the intended final join avoids splitting an annotated CDS; that does not guarantee correct termination or intact regulatory context. Report supplied versus generated sequence, and evaluate completeness and duplicated sequence rather than treating seeded gene reward as newly learned generation.

Interleave anchor/orientation/length strata across training batches. Keep a fixed, independent, comparably stratified validation bank. Native packed generation handles heterogeneous lengths in one ordered call; length-stratum count need not divide the GPU or batch count. Stratify sawtooth metrics by prompt cohort before calling them learning.

## Check sampling and termination

Evo 2 applies temperature, then top-k, then shifted top-p. After top-k renormalization, EOD survives nucleus filtering according to cumulative mass **strictly before EOD**. Check the deployed chain at authentic endpoints, not just raw top-k ranks.

The realized PhiX mixture uses temperature 1, top-k 5, top-p 1.0: nucleus at .999 removed EOD at two endpoints in the earlier multi-anchor experiment. This restores support, not correct placement, and is not a universal policy for other genomes. Audit authentic-EOD frequency and biological length placement separately across all required strata; both can fall while content reward rises.

Generation budget counts assistant tokens; biological length includes prompt bases and excludes control tokens/EOD. PhiX defaults to 6,000 generated tokens and 6,144 context, allowing 6,016/6,024 biological bases for 16/24-base prompts, beyond the 5,800 reward zero. The independent hard length screen is 5,306–5,730 nt; full credit is 5,359–5,550. See the [README's length evidence](../../examples/README.md#current-phix174-gdpo-score-definitions) for viability versus stability and the shaping choices. Both PhiX RL configs enable the missing-EOD whole-reward gate, retaining no-EOD rows in the loss with zero scalar/vector rewards. A wider taper gives shaping credit, not hard acceptance; a flat-zero tail supplies no graded distance signal. Material changes use a fresh result root, leaving existing attempts comparable under their original bounds.

Retain direct authentic-EOD bins: below lower zero, lower taper, full credit, upper taper, and above upper zero. Count caps and non-EOD short outputs separately. Whole-sequence likelihood depends on linear origin, so mixed-origin designs need an origin-normalized method before global likelihood ranking.

The paired calibration sweep's `TARGET_LENGTH=6000` is a biological total, not RL's generated-token budget; it subtracts prompt bases. Set `STOP_ON_EOS=1` when measuring learned termination. The sweep's default `0` suppresses EOD for fixed-length sampling and cannot evaluate terminal placement. The mode is recorded in `sweep_config.json`; changing it needs a fresh sweep directory. Inspect token-derived `stopped_on_eos` and `truncated` in the generation JSONL separately from component scores. Both calibration and deployment caps must lie beyond the scoring zero. Calibration does not replace the deployed full-shape pilot.

## Score and select

Use the same reward definitions, safety policy, asset manifest, host domain, and confirmed host evidence as online RL. Positive and failure controls distinguish biological no-hit from unavailable measurement. Missing configuration, failed tools, and unexplained NOT_RUN need diagnosis; documented inapplicability remains INDETERMINATE and hard-QC-ineligible.

Compare each objective, measurement support, copying, diversity, and uncertainty. The calibration summaries expose nucleotide/safety gates and reward evidence; selection evidence includes uncertainty on aggregate reward and target signal. Report raw and cluster-deduplicated hard-QC yield only from actual final-screening results; missing legacy binary-QC columns do not imply zero yield. For rotation-invariant biological metrics, run reference rotations through the scoring environment. The control checker reports diversity-containing objectives but excludes them and aggregate reward from the invariance comparison: MMseqs preserves supplied starts. Calibration exact-copy flags (`exact_target_copy` and `exact_sft_copy`) compare uppercased sequences at their supplied starts and strands; MMseqs near-copy measurements are separate.

Keep calibration, fixed validation, and final rollout samples separate. Record chosen prompts, seeds, settings, counts, results, and rationale in the runlog. Material sampling changes start a new SFT-anchored attempt; execution-only adaptations can reuse the scientific selection.
