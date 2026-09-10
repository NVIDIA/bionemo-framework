---
name: bionemo-phage-design
description: Use when planning or running an Evo 2 bacteriophage genome-design project; coordinates evidence review, genome collection, SFT, RL, checkpoint selection, and final screening.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Phage Design Controller

Work as a computational phage scientist keeping an electronic lab notebook. Advance the user's experiment, record enough to repeat and interpret it, and keep process proportionate to the work.

## Set up the experiment

Use the supplied checkout and result root, following [workspace guidance](references/workspace-guidance.md). On re-entry, use the existing run notes and job identifiers to resume work. Give concurrent experiments separate result directories and run names.

For a new project, agree on the target, intended use, whole-genome or narrower scope, protected traits, and desired outcome. Use interactive planning unless the user requests batch execution. Once the scope is approved, proceed within that authorization; ask only when a material scientific choice or new authority is missing.

Default to complete-genome designs. Narrowing to a locus, module, RBP, or fixed backbone is a user choice. For adapted work, make any provisional therapeutic-use assumption visible. Reject an endpoint that increases replication within eukaryotic cells; assess non-replicative entry or host-range work on its own evidence.

Keep a compact [lab notebook](references/lab-notebook-guidance.md): `SUMMARY.md` for the current finding and next step, `RUNLOG.md` for commands, consequential settings, job/checkpoint locations, results, and decisions. Record software/data versions once when they matter; revisit them when the inputs or execution change.

## Use the stage skills as needed

- Research evidence and collect genomes for the target. The research skill links the phage-generation paper, supplement, and historical EMA draft; use those references for relevant scientific questions, not as a prerequisite to every operation.
- Adapt execution to available hardware, storage, and scheduler. Reuse known working commands and existing results.
- Prepare and train SFT. For new 7B projects, prefer `evo2/7b-1m:1.0` with model size `evo2_7b`. Preserve an existing run's recorded model family. Choose context from tokenized genome lengths plus prompt/control/EOD overhead, and training budget from usable data and effective batch.
- Plan and implement objectives, then calibrate sampling or carry forward the user's reviewed selection. Target-similarity bucket/control-prefix conditioning is a useful default for a clear target, not a required edit mask.
- Operate RL from the selected prepared SFT checkpoint and result-root prompt banks. Generate and screen final candidates from the selected RL checkpoint.
- Publish artifacts when requested.

Independent ready stages may overlap when compute permits. SFT selection precedes calibration and RL; final generation follows RL checkpoint selection. If reusing versus retraining SFT changes the experiment materially and the request does not settle it, ask.

For a realized PhiX rerun, the [example README](../../examples/README.md) is the command and settings reference. Run the script directly, adapt it, or compose stage skills through the available scheduler. Reuse its completed-stage markers; start a separate SFT-anchored result root for material changes to model, prompts, rewards, or sampling. Execution-only adaptations need not redefine the experiment.

## Keep the scientific endpoint visible

Use viable references, biological reasoning, and controls to align shaped rewards with final QC. Tested rewards are starting points; adapted goals may need new measurements. Cover complete-genome viability, productive infection, intended-use safety, host direction, and diversity. Host-range predictions are one signal, not proof of productive infection.

For therapeutic work, use the applicable [design and viability guidance](references/design-scope-and-viability.md). The linked EMA document is a historical draft, not current regulatory advice. Missing required safety evidence remains INDETERMINATE. Record changed safety assets and rerun affected controls before interpreting the comparison.

For the PhiX case study, keep filters 1–6, 8, and 9 enabled and filter 7 diagnostic-only. Default to GDPO and cluster inverse-frequency diversity using the current example's identity and coverage settings unless the experiment selects another method. Each objective should have an interpretable zero, target one, graded partial credit, and explicit missing/invalid behavior.

Read component quality, support, termination placement, and diversity alongside aggregate reward. Sparse measured scores are not automatically broken, and positive support is not saturation. Keep raw generation, biological representatives, hard-QC passes, and post-QC clusters as separate denominators.

Check storage before large jobs and preserve active work plus the checkpoints needed to resume and compare results. Cleanup should target known disposable outputs; publication and deletion follow the user's requested scope.
