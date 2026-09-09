---
name: bionemo-phage-design-calibrate-rl-sampling
description: Use after selecting an Evo 2 phage SFT checkpoint and defining RL objectives to calibrate prompt format, sampling settings, and train/validation prompt mixtures.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Calibrate Phage RL Sampling

Use the selected SFT checkpoint's actual serialization and the project's scientific endpoint. For a faithful rerun, carry the user's reviewed sampling selection forward; fresh calibration does not override it. For a new selection, compare paired-seed samples across plausible settings and choose a robust quality/diversity region rather than the noisiest maximum.

The [PhiX example README](../../examples/README.md) describes commands, the review stop, selection schema, and resume markers. An explicit `--sampling-selection` is a reviewed override, not the outcome of fresh calibration. If selection is delegated, inspect `calibration/scoring/selection-evidence.csv` and neighboring score/novelty evidence, write `calibration/sampling-selection.yaml`, and continue. No further acknowledgment is needed within the delegated scope.

## Construct prompts

Reconstruct conditioning, orientation, wrappers, tokenization, BOS/EOS, padding, and the continuation boundary from SFT. Use only cues the model saw.

Prompt bases are fixed, not designed. Use the objective plan's exclusions, including circular intervals that wrap the origin. Start with the shortest workable prompt; report its bases and genome fraction rather than scaling length by genome size. Longer prompts need evidence that they help. If a preserved required ORF is never reached from neutral anchors, a minimal start-seeded stratum can be tested alongside an unseeded stratum; record what it fixes.

For confirmed circular or headful/pac genomes, compare alternate rotations and overlap-aware CDS occupancy, including annotated regulatory features. Summed CDS occupancy can exceed 100% for overlapping genes; missing annotations do not establish neutrality. For fixed-end dsDNA, reconstruct the biological termini and packaged/deposited form first, then compare the forward and reverse-complement terminal prompts rather than internal rotations.

Interleave anchor/orientation/length strata across training batches. Keep a fixed, independent, comparably stratified validation bank. Native packed generation handles heterogeneous lengths in one ordered call; length-stratum count need not divide the GPU or batch count. Stratify sawtooth metrics by prompt cohort before calling them learning.

## Check sampling and termination

Evo 2 applies temperature, then top-k, then shifted top-p. After top-k renormalization, EOD survives nucleus filtering according to cumulative mass **strictly before EOD**. Check the deployed chain at authentic endpoints, not just raw top-k ranks.

The realized PhiX mixture uses temperature 1, top-k 5, top-p 1.0: nucleus at .999 removed EOD at two authentic anchor endpoints. This restores support, not correct placement, and is not a universal policy for other genomes. Audit authentic-EOD frequency and biological length placement separately across all required strata; both can fall while content reward rises.

Generation budget counts assistant tokens; biological length includes prompt bases and excludes control tokens/EOD. Put the ordinary cap beyond the length reward's upper zero. PhiX's 5,420 generated tokens yield 5,436/5,444 biological bases for 16/24-base prompts. A wider lower taper gives shaping credit, not hard acceptance. An approved longer-cap/no-EOD experiment uses a fresh result root; it adds no graded distance signal where length reward is already zero.

Retain direct authentic-EOD bins: below lower zero, lower taper, full credit, upper taper, and above upper zero. Count caps and non-EOD short outputs separately. Whole-sequence likelihood depends on linear origin, so mixed-origin designs need an origin-normalized method before global likelihood ranking.

## Score and select

Use the same reward definitions, safety policy, asset manifest, host domain, and confirmed host evidence as online RL. Positive and failure controls distinguish biological no-hit from unavailable measurement. Missing configuration, failed tools, and unexplained NOT_RUN need diagnosis; documented inapplicability remains INDETERMINATE and hard-QC-ineligible.

Compare raw and cluster-deduplicated hard-QC yield, each objective, copying, diversity, and uncertainty. For rotation-invariant metrics, run reference rotations through the scoring environment. Per-row cluster-representative flags are set-relative, so compare their representative count rather than requiring identical flags.

Keep calibration, fixed validation, and final rollout samples separate. Record chosen prompts, seeds, settings, counts, results, and rationale in the runlog. Material sampling changes start a new SFT-anchored attempt; execution-only adaptations can reuse the scientific selection.
