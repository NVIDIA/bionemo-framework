---
name: bionemo-phage-design-operate-nemo-rl
description: Use when launching, monitoring, resuming, or selecting checkpoints from a NeMo-RL Evo2 phage optimization run.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Operate NeMo-RL Phage Training

Run the selected SFT checkpoint with the agreed objectives and sampling settings. Keep useful experimental notes and act within the user's existing authorization. The [PhiX example README](../../examples/README.md) owns the current launch commands, settings, stage markers, and resume procedure.

## Start or resume

- Prepare the SFT checkpoint with `evo2_phage_prepare_sft_checkpoint_for_rl`. Use its direct model-only `iter_*` path for both policy initialization and the fixed SFT KL reference. Schema 2 preserves model object state, including Transformer Engine `_extra_state`; rerunning preparation upgrades a matching schema-1 copy without repeating SFT or calibration.
- Use the result-root train and validation banks. Repeating the example command reuses completed stages and prepares missing downstream inputs; `--resume-from` does not make unfinished stages complete.
- For a compatible continuation, restore the full RL checkpoint and retain its original SFT KL anchor. A deliberate change to rewards, sampling, prompts, or model starts a separate result root. Record model-only recovery as fresh-optimizer continuation, not an exact resume.
- Legacy synchronous GRPO stops at either `max_num_steps` or `max_num_epochs`. Make the epoch budget large enough for the requested number of updates.

## Qualify a new execution shape

Use the [compute guidance](../bionemo-phage-design-adapt-execution/references/compute-guidance.md) for CPU/GPU sizing. Test full-genome batches on the deployed topology: two optimizer updates, validation, then another update. Start with `val_at_start=false`; optimizer state and validation allocations make a one-update test insufficient. Test checkpoint save and a fresh-process reload too.

The native packed adapter should process the whole mixed-length group in one call and return the original row order. With cache offload, look for actual memory release at `finish_generation` and successful reuse on the next rollout. Keep optimizer residency opt-in until a full-shape pilot demonstrates headroom.

Check selected-action generation log-probabilities against full teacher-forced replay before mismatch masking, including EOD and capped rows. Periodic error at KV-page boundaries, rejected rows, nonfinite values, or wrong action masks needs diagnosis. An isolated finite token tail with accepted sequence statistics is an observation to investigate, not a new stopping threshold. Detailed interpretation is in [monitoring guidance](references/monitoring-guidance.md).

## Reward and trajectory semantics

Keep sampled EOD and its log-probability in the action loss; mask synthetic padding and post-EOD suffix. Biological scoring uses only the bases before EOD. A faithfully sampled invalid genome is a negative example, not a broken rollout.

Report three termination outcomes: authentic EOD, capped without EOD, and below-cap without EOD. Use retained tokens and `max_new_tokens`, not allocator-rounded `max_model_len`. Bin authentic stops by total biological length, including prompt bases.

The PhiX configs enable `zero_reward_without_eod` by default with a 6,000-token generation budget. It zeros the **entire** scalar reward or GDPO vector, including safety channels, for no-EOD rows while retaining their actions in the loss (`overlong_filtering=false`). Raw QC and safety results remain diagnostics. An explicit `false` override supports ungated comparisons; keep existing runs on their recorded settings. Watch all-no-EOD and zero-variance prompt groups for lost learning signal. A longer cap adds no distance gradient above a flat-zero length reward, and frequent EOD does not establish correct placement.

Compare configured objectives with emitted scores and positive/failure controls. Separate valid zero or candidate-level safety failure from unavailable measurements. Diagnose missing scorers; keep sparse but measured objectives visible. Record deliberate objective changes and report them in the next useful update.

## Save and select

Use native Megatron-Bridge `torch_dist` checkpoints: `checkpointing.model_save_format: null`, `save_consolidated: false`, Megatron enabled, DTensor disabled. Weights live at `step_N/policy/weights/iter_0000000`. Keep `save_optimizer: true` for compatible continuation; a model-only fallback restores step/dataloader/weights but initializes fresh Adam. Recipe config edits do not require reinstalling NeMo-RL.

The recipe dataset uses task name `phage_qc` regardless of JSONL location. Primary retention uses `val:phage_qc/mean_reward`; TensorBoard normally emits `validation/phage_qc/...`. A path-derived task name such as `rl-validation` points to the generic dataset being used instead.

Retain latest resumable, aggregate-best, and positive strict-endpoint best checkpoints. The supervisor hardlinks selected checkpoints rather than duplicating their payloads. Final selection prefers a strict-positive checkpoint; otherwise it chooses the best interior aggregate checkpoint and reports that strict qualification was not achieved.

Follow [monitoring guidance](references/monitoring-guidance.md) for scientific trends, W&B cluster-size histograms, and runtime diagnosis. Record commands, consequential settings, job/checkpoint paths, validation results, and decisions in `RUNLOG.md`; summarize the current finding and next step in `SUMMARY.md`. Use optimizer step for comparisons: W&B's history cursor is a logging counter, and its service can fail while training continues.
