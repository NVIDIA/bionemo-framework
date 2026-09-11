# RL monitoring guidance

Use a job facility that survives the chat session. Record its identifier and log location, check startup, then observe at the cadence of meaningful progress or the next validation/checkpoint. Use a harness scheduler, a scoped cron task, or an attached waiting process when available. Reattach after reconnecting. Repeated unchanged polls add little to the runlog.

## Read the experiment

Read training rollouts and the fixed validation bank together. The bank provides longitudinal comparison; different seeds or prompts do not make it a biological-label holdout. Track:

- reward level and variability, component means, support, and hard-QC yield;
- measurement availability, safety outcomes, and reasons for missing results;
- authentic EOD, cap exhaustion, and non-EOD short output separately;
- authentic-stop biological length bins, prompt strata, copying, and diversity.

Positive support is not a full-credit score. More gene content or a higher aggregate can coexist with worse termination. When diversity is gated by length or safety eligibility, inspect diversity among eligible rows as well as its all-row mean. Check prompt composition before attributing cycles to learning.

Use the native W&B histogram at `train/phage_qc/mmseqs_cluster_size` or `validation/phage_qc/mmseqs_cluster_size` for cluster-size distributions over optimizer steps. Color represents cluster count: one observation per unique cluster, with unclustered rows excluded. Validation pools scoring-batch distributions without reclustering. The old `mmseqs_cluster_size_histogram/size_*` scalar family is no longer emitted; TensorBoard still receives scalar cluster summaries.

Use comparable windows rather than SFT-style patience. At validation every ten steps, roughly ten banks (about 100 updates) is a useful horizon for noisy RL, not an automatic countdown. A recovered excursion or a plateau below an earlier peak is not by itself a reason to stop. Sustained deterioration across supported components and training rollouts warrants diagnosis and a checkpoint decision within the agreed experiment budget.

Low measured safety scores and faithfully sampled invalid genomes are learning outcomes. Missing artifacts, unexplained NOT_RUN, or failed enabled scorers make the evidence uninterpretable and need repair. Record a concise continue, diagnose, stop, or restart decision at useful scientific boundaries; ask only when the next action needs a genuinely new user choice or authority.

## Diagnose replay or memory problems

For selected-action replay, inspect token deltas, the configured sequence-level statistic, and rejected rows before masking. Include terminated and capped outputs. Group errors by absolute token position modulo the KV-page size: the known rollover defect produced a sharp 256-token pattern despite modest median error. Verify two generation/replay/refit cycles at the actual topology, precision, graph scope, and batch shape.

A single finite tail away from a boundary, with ordinary population statistics and no rejected row, merits a recurrence check. The sequence guard is not a per-token threshold. Nonfinite values, wrong masks, rejected rows, or persistent page-phase elevation call for diagnosis.
The finite surviving subset does not qualify a broken pilot or justify accepting its updates.

Separate cold setup/capture from steady generation and decode using the emitted `evo2_*_completion_tokens_per_s` metrics. Persistent graphs can be reused while bound storage stays valid; parameter/buffer rebinds require replica-wide TP/PP/CP recapture, and quantized graphs recapture after refit. Use `EVO2_EXACT_PHASE_EVIDENCE=1` when synchronized timing or allocator evidence is needed.

For memory failures, check PyTorch allocation/reservation, other device processes, and host RSS. An already-batched QC RPC needs `max_concurrency=1`; extra actor threads caused a host-memory staircase that surfaced during checkpoint saving. CPU QC actors should hide CUDA and set `PHAGEHOSTLEARN_ESM_DEVICE=cpu`, leaving internal scorer CPU parallelism intact.

With native cache offload, confirm state is deallocated after generation, restored before reuse, and graphs recaptured when storage changes. The default-false `preserve_optimizer_state_during_generation` option avoids Adam transfers but still offloads gradients; test its actual peak memory through train→generation→train before enabling it.

## Keep useful results

Preserve the latest resumable checkpoint, aggregate-best, and positive strict-endpoint best independently. A flat-zero strict metric should not discard useful shaping progress; a high aggregate is not a biological pass. Record the selected checkpoint and why. Use durable training logs/checkpoints to distinguish a tracker outage from a stalled trainer.
