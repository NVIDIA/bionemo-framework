# PhiX174 whole-genome example on 8×H100

[`phix174_8xh100.sh`](phix174_8xh100.sh) prepares public inputs, performs supervised fine-tuning
(SFT), and optimizes the model with [GDPO](https://arxiv.org/abs/2601.05242), a multi-reward
reinforcement learning (RL) method. It then generates a rollout of 1,000 complete genomes and
applies the current safety and PhiX174 design screens. This reference launcher has been tested on an
8×H100 server. Other topologies may require adjusted settings. The `bionemo-phage-design` skill can
adapt the settings and help run and monitor the job.

This example uses filters 1–6, 8, and 9 from
[Figure 2H](https://www.science.org/doi/10.1126/science.aec2657#F2), following Samuel King's
recommendation (personal correspondence). RL represents their measurable constraints as separate
graded or categorical objectives; reward full-credit regions and final hard-pass thresholds are not
interchangeable, and batch diversity is not a per-genome hard gate. See the
[PhiX174 GDPO score definitions](#current-phix174-gdpo-score-definitions) section below for
the exact reward, checkpoint-selection, and final-filter relationships.

The end-to-end run takes approximately 4 days on a server with 8 H100 GPUs, and uses 1.5TB of storage.

## Latest completed PhiX174 result

The 8×H100 rerun completed on 2026-08-24:

| Final-rollout denominator                                  |       Count |
| ---------------------------------------------------------- | ----------: |
| Raw generated and SFT-likelihood scored                    |       1,000 |
| Representatives retained before hard QC                    |       1,000 |
| Submitted to safety / excluded by pre-safety QC            |     991 / 9 |
| Safety PASS / FAIL / INDETERMINATE                         | 989 / 0 / 2 |
| Safety-PASS target hard-QC representatives                 |         513 |
| Post-QC 99%-identity clusters and accepted representatives |         511 |

The denominators proceed from raw generation through biological deduplication, safety screening,
and target hard QC, then post-QC clustering. Likelihood is a within-protocol ranking signal; these
computational candidates are not evidence of bootability or wet-lab safety. See the
[case-study notes](../skills/bionemo-phage-design/references/case-study-results.md) for historical context.

The current launcher starts a new `results/phix174-8xh100-origin` result root. Its reward and
sampling settings have changed since that completed run; the table is historical evidence.

## Quick start

Run from `recipes/evo2_phage_gen`. The PhiX follow-up uses the published-lineage `7b-base` checkpoint
with the current origin-only defaults. Supplying `--sampling-selection` explicitly skips the
fresh-calibration review stop:

```bash
./.ci_build.sh
tmux new -s phix174-e2e
./examples/phix174_8xh100.sh \
  --model-variant 7b-base \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-8xh100-origin"
```

The build reuses the native Torch/CUDA/Transformer Engine stack already in the
NVIDIA PyTorch container. The launcher sources the resulting recipe-local
`.ci_test_env.sh`; for standalone commands, source that file yourself.
Use the base image named in the recipe [Dockerfile](../Dockerfile), currently
`nvcr.io/nvidia/pytorch:26.07-py3`, with a compatible host driver. Build a fresh recipe environment
when changing images; do not reuse compiled extensions or older image-specific dependency pins.
`.ci_test_env.sh` selects the recipe-installed cuDNN when present so the driver process and Ray
workers use the same runtime. Before training on a new image/node combination, verify CUDA
initialization, NCCL collectives, and the required subquadratic kernel on every worker.

The RL configs select cuDNN fused attention with
`policy.megatron_cfg.attention_backend=fused`. This avoids the FA4/CuTe backward stall
reproduced on the tested 26.07 H100 stack; environment-only `NVTE_*` selectors are not
sufficient because the Evo2 provider can reset them. This setting applies to RL policy
and reference computation, not the native packed-generation adapter. Qualify full updates,
validation and checkpoint reload when changing the runtime or attention backend.

For a fresh PhiX experiment, use the trained-further 7B-1M model and a new result root:

```bash
./examples/phix174_8xh100.sh \
  --model-variant 7b-1m \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-7b-1m"
```

`7b-base` is the default and matches the published Microviridae lineage. `7b-1m` selects
`evo2/7b-1m:1.0` and provider `evo2_7b`. Checkpoint lineage is authoritative, and the script
refuses to change model families within an existing result root.
The two-character SFT conditioning prefix remains input context but is excluded from next-token
loss; the biological sequence immediately after it remains supervised.

## Common operations

```bash
# Inspect commands without downloads or GPUs.
./examples/phix174_8xh100.sh --dry-run --result-root /tmp/phix174-plan

# Prepare public inputs, tools, databases, and controls only.
./examples/phix174_8xh100.sh --prepare-only --result-root "$PWD/results/phix174-8xh100-origin"

# Resume the RL stage of the 7b-base run.
./examples/phix174_8xh100.sh --resume-from 40 --model-variant 7b-base --result-root "$PWD/results/phix174-8xh100-origin"

# Use an explicitly reviewed sampling selection, without blocking if the automatically identified
#  top setting differs. In practice automatic selection can be noisy, and these settings should work
#  well for phix174.
./examples/phix174_8xh100.sh --sampling-selection examples/default-sampling-selection.yaml --result-root "$PWD/results/phix174-8xh100-origin"
```

Completed stages and substages are skipped. Reuse the same result root and sampling selection when
resuming; `--resume-from` selects where checking resumes but does not create missing state or
completion markers. A material prompt, reward, sampling, or model change should start a new result
root. Stage 20 records its real-data restart smoke separately as `stages/20-sft-smoke.done`; it
validates and reuses a complete converted base checkpoint instead of redownloading or reconverting
it. Stage 40 likewise keeps `stages/40-pilot.done`, `stages/40-pilot-reload.done`,
`stages/40-pilot-check.done`, and `stages/40-rl.done` distinct, and reuses only a validated schema-2
optimizer-free prepared SFT checkpoint. Do not create a marker unless its operation is known to
have succeeded. Before new calibration scoring, RL environment checks/training, or final
Arc screening, the launcher regenerates the derived Arc pipeline from the current
maintained patch. This also applies when starting directly at stage 40 or 50;
completed scientific outputs remain controlled by their existing stage markers.

### Optional W&B logging

Authenticate with `wandb login` (or provide `WANDB_API_KEY` through a secret manager), then opt in:

```bash
./examples/phix174_8xh100.sh --wandb --wandb-entity YOUR_ENTITY \
  --wandb-sft-project evo2-phage-design-sft \
  --wandb-rl-project evo2-phage-design-gdpo \
  --result-root "$PWD/results/phix174-8xh100-origin"
```

The project flags are optional and show their defaults. Run names are derived from the result-root
name and model variant. SFT has its own optional W&B run. The GDPO W&B run is not created until the
full stage-40 training starts: calibration and the pilot stay offline, while the full run honors
the entity, project, and name overrides. The launcher gives W&B initialization 300 seconds; set
`WANDB_INIT_TIMEOUT` to override it. Never pass an API key as a launcher argument. Completed stages
are not uploaded retroactively.

### Choose sampling settings

[`default-sampling-selection.yaml`](default-sampling-selection.yaml) contains the current PhiX
defaults and is the schema for a custom selection. Passing `--sampling-selection` is an explicit
override, not a choice derived from the fresh calibration evidence. The defaults use temperature
1, top-k 5, and top-p 1.0, which disables nucleus filtering while retaining the complete top-5
support so the length reward can teach termination. When qualifying another cutoff, reproduce the
deployed temperature, top-k, then shifted top-p chain: after top-k renormalization, EOD remains in
support according to the cumulative mass strictly before EOD. Verify that deployed support includes
EOD at the authentic boundary for every prompt stratum; checking its raw rank before top-p is not
sufficient. That verifies reachability only: fixed-bank monitoring must track authentic-EOD
frequency and full-credit length placement separately, since both can fall while aggregate or
gene-content reward rises.
The default 6,000-token generation budget permits 6,016/6,024 biological nt across the prompt
mixture, 216/224 nt beyond the length reward's 5,800-nt upper zero, so a no-EOD rollout receives
no length credit. The broad 3,000–5,359-nt lower taper reaches initially short rollouts; full
credit spans 5,359–5,550 nt and the separate hard length screen accepts 5,306–5,730 nt.
[Russell and Müller (1984)](https://doi.org/10.1128/jvi.52.3.822-827.1984) recovered insertion
mutants up to 5,730 nt but reported instability above 5,550. The 16 genomes in
[King et al.'s released viable FASTA](https://github.com/ArcInstitute/evo2/blob/main/phage_gen/data/viable_generated_phage_genomes.fasta)
span 5,349–5,654 nt; four exceed the former 5,493-nt hard ceiling. The 5,800 zero is an engineering
shaping choice, not a measured packaging limit. This screen admits longer viable examples without
claiming that every passing genome is viable or stable. Rokyta et al.'s
[5,486–5,577-nt isolates](https://doi.org/10.1128/JB.188.3.1134-1142.2006) belong to the G4-like
clade; their PhiX174/S13-like isolates span only 5,386–5,387 nt. These future defaults do not change
the thresholds or scientific interpretation of existing runs. In the paired calibration
sweep, `TARGET_LENGTH=6000` counts total biological bases and subtracts each
prompt length from the generation budget, keeping paired cells at one ceiling beyond the reward
zero. RL instead allows 6,000 **generated** tokens. Both use 6,144 context; the calibration sweep
does not substitute for the deployed full-shape pilot.

### Prompt origin

Calibration, RL training, fixed validation, and final generation all default to **reference
coordinate 1**, with 16- and 24-nt prefixes and the `+~` conditioning tag. Here `origin` names the
deposited sequence start, not the biological replication origin. Prompt bases remain fixed,
including the short segment of origin-spanning coding sequence; the remaining bases are generated.

This choice matches the observed start context of related assemblies in the
[released Microviridae SFT corpus](https://zenodo.org/records/17101843). In a September 16, 2026
audit of all 14,466 records, all eight records explicitly named PhiX174 started with the canonical
16-mer. Exact start matches to the four previously tested PhiX prefixes were:

| PhiX reference coordinate (1-based) | Exact 16-base prefix | Records starting with it |
| ----------------------------------- | -------------------- | ------------------------ |
| 1                                   | `GAGTTTTATCGCTTCC`   | 20                       |
| 2,387                               | `GTTTAATCATGTTTCA`   | 0                        |
| 3,918                               | `CCGTCAGGATTGACAC`   | 1 (S13, `M14428.1`)      |
| 3,973                               | `GCTTTTTTATGGTTCG`   | 0                        |

All processed sequences matched their raw counterpart after removing the two conditioning
characters: the released preprocessing added no rotation augmentation. The maintained SFT path
also preserves sequence starts; its circular-equivalence grouping prevents split leakage and
does not augment training with rotations. Thus circular biology alone does not establish that
an internal prefix is a familiar beginning-of-genome context for this model. We use the supported
start to give RL a better initial gene-generation signal.

These are exact-prefix counts before held-out splitting, not an alignment-based census of all
relative start sites or verified exposure of a particular checkpoint. They support a concentrated
start convention in this corpus, not a universal submission rule. For reproducibility, the processed
file is `microviridae_sft_training_data_processed.fna`, SHA-256
`0b86cb2ecc6f742ad96ba47d9a08fd3060af9880cb8ef4d33b886949a1ec4b1d`.
Before adapting rotated prompts to another target, compare related assemblies' start coordinates
and orientations, submission/annotation conventions, and the actual SFT representation. The
[collection guidance](../skills/bionemo-phage-design-collect-genomes/references/collection-guidance.md)
describes that check. Circular ORF handling and rotation-invariant biological scores remain appropriate
for PhiX regardless of prompt placement.

Each 768-rollout GDPO update uses two prompt records × 384 generations: one group for the
16-base origin prompt and one for the 24-base origin prompt. GDPO normalizes each objective
within identical prompt token sequences, so repeated copies of a prompt record do not create
independent normalization groups. Training metrics `reward_prompt_group_count` and
`reward_prompt_group_size_min` / `reward_prompt_group_size_max` use the actual prompt tokens;
the default update should report 2 groups and minimum/maximum sizes of 384. This explicit layout preserves the two 384-member groups of
the previous balanced 16-record × 48-generation layout. Diversity clustering pools eligible
genomes from both prompts in the scoring batch because they share one design goal.
The training and independent fixed-validation banks
each contain 96 records, balanced 48 per length. The 1,000-design final rollout uses 500 prompts
per length, all starting at coordinate 1.
Each GPU generates its shard in packed batches of 96 by default; set
`FINAL_PROMPT_BATCH_SIZE` only when qualifying a different device profile.
The previous 5,420-token/5,632-context 768-point setting completed three 8×H100 rollout/QC/replay/update
cycles with zero replay masks, including validation after update two, a post-validation update, an optimizer-inclusive checkpoint,
and a fresh-process exact reload. Warm update two took 492.65 seconds for generation including
233.78 seconds of scoring, 37.74 seconds for policy/reference replay, and 54.60 seconds for the
policy update. A 1,024-point attempt failed recurrent-state allocation before decode, so it is not
a supported one-wave setting.
Legacy synchronous GRPO is bounded by both steps and epochs; this 96-row bank supplies 48 updates
per epoch, so the configured 500 epochs safely exceeds the requested 500-step ceiling.

The launcher sets `max_model_len` to the smallest 256-token boundary covering the longest selected
prompt, its two-token `+~` prefix, and `max_new_tokens` (6,144 with the defaults), rather than
allocating the unused remainder of the 10,240-token SFT context. Native RL trajectories retain the
first sampled EOD and its log-probability, mask only synthetic padding, and exclude EOD and any
post-EOD physical samples from biological QC. Filtered policy replay also keeps each sampled action
in a normalized target-preserving support; generation-versus-replay error telemetry remains enabled.
Qualification reports authentic EOD, capped-without-EOD, and below-cap-without-EOD as three
exclusive outcomes. The `phage_qc/termination/*` scalars retain rates for those outcomes
and place authentic-EOD genomes into below-lower-zero, lower-taper, full-credit, upper-taper, and
at-or-above-upper-zero bins. Outcome rates use all sequences; length-bin rates use authentic-EOD sequences.

Both PhiX RL configs enable `env.phage_qc.zero_reward_without_eod=true` by default. It
keeps every row and sampled action in the RL loss, but assigns an exact-zero scalar reward—or an
all-zero GDPO objective vector—unless token metadata records an authentic EOD. Raw safety and QC
measurements remain truthful diagnostics, but cannot reward the invalid row. This is the inverse of overlong loss filtering:
caps remain negative examples instead of disappearing from the advantage calculation. The default
`max_new_tokens=6000` and `max_model_len=6144` leave no-EOD caps at 6,016/6,024 nt. Length reward
is flat zero above 5,800 nt: that tail supplies no graded distance signal. The longer generation
budget and reward gate are separate settings, both enabled in the future-run defaults.
Set `env.phage_qc.zero_reward_without_eod=false` explicitly for an ungated comparison;
keep `grpo.overlong_filtering=false` so no-EOD examples retain their loss contribution.
The gated runs support improved termination frequency, not yet reliable target-length
placement or viability. This default does not change existing runs or rescore old banks.
Monitor `termination/no_authentic_eod_prompt_group_count` and
`reward_zero_variance_prompt_group_count`: an all-no-EOD group has no termination advantage and can
make the gate self-reinforcing. Use a fresh result root for this material reward and sampling-budget
change.

The policy defaults to global batch 768, candidate training microbatch 8, validation 96, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The standard pilot disables initial validation,
trains for two updates, validates, and then completes a third update. Adam state first materializes
during update one, and the post-validation update detects memory or graph state retained by
validation, so a one-update pass or terminal validation does not establish steady-state capacity.
MBS8 is the qualified 8×H100 default, not a measured steady-state maximum. The 6,000-token,
6,144-context missing-EOD experiments also completed repeated updates, periodic validation,
optimizer-inclusive saves and full-state recovery; this is capacity evidence, not qualification
of the newly widened reward or a claim that the optional gate improves biology. Set
`RL_TRAIN_MICRO_BATCH_SIZE` to qualify another candidate through the same
pilot without changing global batch or rollout size. For Evo2 7B at TP1, its largest local GLU output
is `8 × 6,144 × 11,008 = 541,065,216` elements, below signed-int32 indexing; the launcher
rejects larger resolved shapes before worker allocation. TP2 halves the local FFN width and DP only
distributes the global batch, so neither replaces a full-shape memory and replay qualification on
the deployed H100 topology. Keep `logprob_batch_size=1` as the conservative independent setting.

`RL_PROMPT_BATCH_SIZE` controls the packed decode group size (default 96, the DP8-local share of the
768-rollout global batch); set a larger value only on a qualified device profile with enough cache
capacity. The candidate MBS8 is paired with explicit cache offload in this 8×H100 launcher. The adapter
deallocates paged-KV and Hyena state before policy training, then restores them
and recaptures graph runners because their buffer addresses changed. Non-persistent cache settings
must reach MCore's `InferenceConfig`, and a release that leaves tensor state allocated fails rather
than proceeding toward a later OOM. The pilot also requires a direct `finish_generation` observation
that PyTorch allocated memory decreased; accepting an offload config is not release evidence.
Persist-cache deployments need lifecycle-aware capacity qualification. Initial validation left
about 12.55 GiB more allocated (152.25 versus 139.70 GiB)
in early single-GB300 trials, and roughly 29.9 GiB of device-wide framebuffer use made their
capacity failures inconclusive. A corrected `val_at_start=false`, offload-cache run at MBS32 then
completed two full optimizer-resident rollout/update cycles despite that external baseline. The
restarted TP1 production trajectory subsequently completed steps 11-13 at MBS32 with clean 256-row
replay audits while still carrying roughly 42-44 GiB of external device use. This establishes MBS32
for the corrected no-leading-validation/offload profile, but it is not a clean-device ceiling. On
cycle two, `finish_generation` reduced PyTorch allocation from 202,501 to 75,431 MiB and device use
from 243,653 to 114,613 MiB; replay over 921,250 active tokens had no masks, nonfinite values,
large deltas, or page-phase elevation. This qualifies MBS32 only for that measured GB300 profile
and directly validates native cache release; persist-cache MBS32 remains unproven. MBS64 exceeds
the TP1 signed-int32 GLU limit, making MBS32 the largest robust candidate on that path. Do not
transfer it to H100: start with training, reach optimizer steady state, run a scheduled validation,
and then complete a subsequent update on the deployed topology.
Expandable segments fix fragmentation, not retained-state capacity.

GDPO checkpoints preserve optimizer state by default. An earlier 7B TP1 failure at the first
optimizer save was traced to an already-batched QC actor configured with `max_concurrency=1000`:
it created 1,052 threads and retained roughly 4.5 GiB of host RSS per update, reaching about 108 GiB
by step 20. The checkpoint allocation was the final trigger, not the leak. With
`max_concurrency=1`, the actor remained near 16.3-17.6 GiB and 52 threads; a 74 GiB full-state
checkpoint finalized atomically in 82.25 seconds, peak policy RSS was 286.1 GiB, at least 117.2 GiB
of host memory remained available through overlap with the next rollout, and the next policy update
completed. The stage-40 pilot launches a separate process against its step-3 checkpoint and requires
restored step, dataloader, weights, and optimizer without a fresh-Adam warning before full training.
If fixed-actor host capacity cannot support full state, explicitly set `save_optimizer=false`; the
same check accepts that model-only recovery only with the expected fresh-Adam diagnostic and records
that it is not trajectory-identical.

Optimizer state is offloaded during generation by default. On a device with measured HBM capacity,
`policy.generation.mcore_generation_config.generation_adapter_config.preserve_optimizer_state_during_generation=true`
avoids that per-step CPU round trip while still offloading gradients. Qualify it in a disposable
full-shape train→generation→train pilot using PyTorch allocated/reserved memory and observed peaks,
not `nvidia-smi` aggregate accounting.

To stop after the sweep and scoring, before prompt banks or RL are created, run:

```bash
./examples/phix174_8xh100.sh --calibrate-only --result-root "$PWD/results/phix174-8xh100-origin"
```

Inspect `results/phix174-8xh100-origin/calibration/scoring/selection-evidence.csv` and its neighboring
score/novelty artifacts. Prefer eligible settings with working metrics, useful hard-QC signal,
low copying, diverse outputs, and a stable quality-diversity plateau. An agent may perform this
review and write the custom choice when the user delegates it. Copy the example YAML, including its
named `prompt_anchors`, then continue without repeating the completed sweep:

```bash
cp examples/default-sampling-selection.yaml /tmp/phix174-sampling.yaml
# Edit /tmp/phix174-sampling.yaml, then:
./examples/phix174_8xh100.sh --resume-from 30 --sampling-selection /tmp/phix174-sampling.yaml --result-root "$PWD/results/phix174-8xh100-origin"
```

The script validates and records the file as `calibration/sampling-selection.yaml`. Do not replace
that canonical selection after RL has begun; a material change should use a new result root.

Calibration scoring uses the same sequence-safety asset manifest, policy, bacterial host domain,
and confirmed versioned PhiX host evidence as online RL. Sampling cells fail closed when required
external or safety evidence is unexplained or unavailable; exact documented biological
inapplicability remains candidate-level `INDETERMINATE`, not a safety `PASS`.

## Script workflow and outputs

| Stage | Work                                                                                        |
| ----- | ------------------------------------------------------------------------------------------- |
| 00    | Download inputs, tools, databases, and run controls                                         |
| 10    | Safety-screen inputs and build leakage-controlled SFT splits                                |
| 20    | Train, select, and evaluate SFT                                                             |
| 30    | Calibrate generation and materialize RL prompt banks                                        |
| 40    | Prepare SFT for RL, run the three-step pilot/reload/check, train GDPO, and select RL        |
| 50    | Generate, SFT-score, deduplicate, safety-screen, hard-QC, cluster, and report 1,000 genomes |

The result root is the computational notebook. `RUNLOG.md` records commands and liveness;
`settings.json` records key settings; `SUMMARY.md` and `rollout/final-designs.json` reconcile
selected checkpoints, safety outcomes, QC denominators, clustering, and accepted candidates.
Final-design reports retain schema 2 and `counts.post_qc_99pct_clusters`. Exact thresholds
are recorded in `evidence.post_qc_clustering.mmseqs`; current coverage is 95%, distinct
from historical 80%-coverage reports at the same 99% identity.
The terminal and `RUNLOG.md` end with `RUN COMPLETE`, `RUN PAUSED`, or `RUN FAILED` plus stage
progress; only `RUN COMPLETE` denotes successful completion of the requested invocation.
Final likelihood scoring runs before deduplication and uses the validated direct model path in
`rl/sft-checkpoint/preparation-manifest.json`, as does RL. The original full-state SFT checkpoint
remains available for exact SFT resume. The `rollout/accepted_candidates.fasta`
contains the final filter-passing, deduplicated candidates for further analysis. When its scores are
informative, not strongly length-associated, and drawn from one circular origin, this FASTA may be
ordered by mean per-nucleotide likelihood under the selected SFT model. Mixed-origin runs retain
generation order because whole-sequence language-model likelihood depends on the linearized origin;
the scores remain recorded as diagnostics.
The final rollout and the inner GDPO rollout both use packed dynamic prefill and batched recurrent
decode for medium/long generation. GDPO assigns 96 requests to each of eight data-parallel replicas
for its 768-sequence step and retains sampled EOD actions so the length reward can teach termination.
Final selected-SFT
likelihood uses packed prediction for ragged batches while preserving FASTA record mappings. NeMo-RL's
separate `policy.sequence_packing` option remains disabled until its gradient-bearing loss/backward
path is independently qualified; it does not control rollout packing.
Each shard keeps its decode rows logically active and captures only the physical full/remainder
batch shapes it actually runs. Graph reuse requires stable registered model storage; an unexpected
parameter or buffer address change on any TP/PP/CP rank forces replica-wide recapture without
crossing DP replicas. Qualify at least two rollout/offload-refit cycles. Report cold end-to-end,
setup-free generation, and steady-decode throughput separately; set
`EVO2_EXACT_PHASE_EVIDENCE=1` only for synchronized phase and allocator diagnostics.
[Black et al., “Quantifying evolutionary novelty and design efficiency in generative genome
design”](https://www.biorxiv.org/content/10.64898/2026.06.12.731871v1.full) found that Evo 2
likelihood predicted experimental viability within a previously filtered PhiX174 dataset, but the
score used here remains a within-protocol ranking signal rather than a bootability probability or
transferable threshold.

Before GDPO, the exact configured environment scores the reference at coordinate 1. An explicit
custom selection with additional anchors also checks those reference rotations for identical
reward, filter, and measurement-support outcomes. Arc hard QC removes ORFipy calls beginning
wholly inside its appended pseudocircular tail, while retaining cross-origin ORFs, so tail length
cannot create rotation-dependent duplicate genes.
Raw endpoint-local DUST fractions can differ by linear origin, but the tested PhiX rotations have
the same intrinsic reward and pass outcome; the exact control excludes per-row cluster-deduplicated
representative flags because those are set-relative rather than properties of a rotation.

The reference topology is eight H100 80 GB GPUs. `NUM_GPUS` defaults to 8. The launcher detects
CPU availability without inheriting tool-specific OpenMP limits and advertises at most 150 slots to
Ray, leaving host headroom on the 160-CPU H100 shape; `NUM_CPUS` remains an explicit override.
Stage-40 reward phases run sequentially: external QC uses 64 jobs × 2 MMseqs threads and grouped
diversity uses 16 jobs × 8 threads, so each phase stays within 128 worker threads. SFT tensor
parallelism defaults to 1 on a single GPU and 2 otherwise. The discarded, metadata-heavy external-QC
tree defaults to node-local `${TMPDIR:-/tmp}`; set `RL_EXTERNAL_QC_WORK_ROOT` when the node's scratch
lives elsewhere. On launcher failure, retained Arc evidence is packed under
`failure-artifacts/` in the result root. Durable logs, checkpoints, safety results, and final
artifacts remain under the result root. Override
`SFT_TENSOR_PARALLEL_SIZE` only after a full-shape smoke test. Packed dynamic prefill interleaves
the selected prompt lengths inside each deterministic GPU shard, so length-stratum count does not
need to divide a microbatch or the GPU count. `SFT_MAX_STEPS` defaults to 12,000 as a safety ceiling;
lower it only to a persisted validation decision boundary when cleanly completing an evidence-stopped
compatible run. `train_evo2` resumes step, optimizer, and RNG state from an existing result checkpoint;
the `--finetune-ckpt-dir` base is used only when no run checkpoint exists. Use tmux or a scheduler for
the long stages.

BF16 remains the portable default. After H100/H200 qualification, pass
`--hopper-fp8-inference` to use regular FP8 across compatible 7B linears for calibration, rollout,
and likelihood scoring without changing GDPO training precision. Vortex-style FP8 is separate.
Decode batches divisible by eight avoid regular FP8's alignment fallback.

## Current PhiX174 GDPO score definitions

This is the human-readable contract for the 14 objectives in
`configs/gdpo_phage_megatron.yaml`. It is also the worked example for the run-specific
`artifacts/RL_SCORE_DEFINITIONS.md` that an agent writes when designing or changing objectives;
the E2E shell script does not generate that artifact. These thresholds reproduce the current
PhiX174 computational profile, not universal phage-design optima or evidence of bootability.

GDPO receives each objective row below as a separate `[0, 1]` objective. The scalar `weight_*`
settings use 1 for each enabled scalar component. They affect scalar GRPO rewards and
scalar summaries, while GDPO independently standardizes each configured objective within
its prompt group and sums them with coefficient 1. A zero scalar weight does not disable
a GDPO objective; the `gdpo_objectives` list selects those columns. The first 11
objectives are forced to zero unless the sequence has an exact sequence-safety `PASS`; the three
safety objectives remain unmasked so failures still provide learning signal. Missing, invalid, non-finite,
or unavailable measurements map to zero when scoring returns a row. Configured Arc, DUST, or
diversity-command failures can instead fail the scoring batch rather than fabricate a biological
score. Final checkpoint selection prefers a positive `all_objectives_max_score_rate`
(every configured gated objective reaches 1 on the same sequence), then falls back to
the best interior aggregate checkpoint. Final screening determines QC acceptance.

### Objective implementation map

The ordered objective inventory and reward-column wiring live in the
[GDPO configuration](../configs/gdpo_phage_megatron.yaml). The environment validates and
assembles that matrix in
[`gdpo_objective_scores_from_scored`](../src/bionemo/evo2_phage_gen/nemo_rl_env.py), including
the exact-safety mask. The implementations for the individual terms are:

| Objective                  | Primary implementation                                                                                                                                                                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `valid_nt_chars`           | [`has_valid_nt_chars`](../src/bionemo/evo2_phage_gen/qc.py) and [`add_nucleotide_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                           |
| `genome_length`            | [`score_genome_length`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                                                                                              |
| `gc_content`               | [`calculate_gc_content`](../src/bionemo/evo2_phage_gen/qc.py) and [`add_nucleotide_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                         |
| `nt_homopolymer`           | [`calculate_nt_homopolymer_len`](../src/bionemo/evo2_phage_gen/qc.py) and [`add_nucleotide_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                 |
| `dustmask_end`             | [`calculate_dustmasker_metrics`](../src/bionemo/evo2_phage_gen/qc.py) and [`add_nucleotide_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                 |
| `tropism`                  | [`smooth_protein_match_integrity`](../src/bionemo/evo2_phage_gen/protein_evidence.py), [`summarize_smooth_reference_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py), and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py)             |
| `required_genes`           | [`summarize_required_gene_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_required_gene_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                           |
| `synteny`                  | [`score_function_matches` / `smooth_protein_match_integrity`](../src/bionemo/evo2_phage_gen/protein_evidence.py), [`score_smooth_synteny`](../src/bionemo/evo2_phage_gen/protein_evidence.py), and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py) |
| `gene_a_origin`            | [`score_gene_a_origin`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                     |
| `average_protein_identity` | [`summarize_best_hit_aai`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`score_aai_novelty` / `score_aai_evidence`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                       |
| `mmseqs_cluster_diversity` | [`add_mmseqs_cluster_diversity_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                                                                             |
| `safety_amr`               | [`run_amrfinder_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                             |
| `safety_toxin`             | [`run_toxin_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                 |
| `safety_lysogeny`          | [`run_phrogs_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                |

For module responsibilities and reusable scoring entry points, see the [reward API reference](../skills/bionemo-phage-design-implement-rl-objectives/references/reward-api.md).

The `synteny` objective uses protein/function matching, circular order, and copy
penalties. Arc also has a separate **start/stop-codon landmark score**, named
`genetic_architecture` upstream. That score has no RL objective here: final
screening enables its composite keep range, and filter 7 uses its whole-genome
score in a separate diagnostic. The upstream stage named
`genetic_architecture_visualization_and_synteny_filtering` instead produces protein
synteny, AAI, and required-gene measurements. See the
[configuration distinction](../configs/README.md#synteny-and-arcs-codon-landmark-score)
for the exact flags and their roles.

### How rewards, gates, and selection differ

- **RL objectives** are the rows below. Their `[0, 1]` values shape training; full credit is a target,
  not an automatic final acceptance boundary.
- **Online measurement** disables Arc's length prefilter so a tool-safe length outlier still receives
  independent ORF, protein, and synteny measurements. Length remains its own graded reward and
  final acceptance gate.
- **Configured objective metrics** report `gdpo/{name}_{mean,std,nonzero_rate,max_score_rate}`
  after safety and EOD gates. `max_score_rate` means a score of exactly 1, not a hard-QC pass.
  `all_objectives_max_score_rate` is the fraction of all rollout sequences reaching 1 on every
  configured objective. The scalar GRPO path reports positive-weight components under `component/`.
  Set `env.phage_qc.log_by_prompt_nt_length: true` for per-length breakdowns; the default is false.
- **Checkpoint retention and selection** keep the latest resumable checkpoint, aggregate best,
  and the best positive `all_objectives_max_score_rate` independently. Selection prefers the
  latter, breaking ties by aggregate reward and then step. If no sequence maximizes all objectives,
  it selects the best interior aggregate checkpoint and records `has_max_score_sequences: false`.
  This is training-score attainment; final acceptance still requires the screening stage below.
- **Final per-genome QC** uses exact safety `PASS` plus the Arc target-profile waterfall: A/C/G/T
  only; length 5,306–5,730 nt; GC 30–65%; homopolymer ≤10; a PhiX G hit at 60–100% identity with ≥0.95 query and target
  coverage; mean per-ORF PHROGs member identity 0–95%; all nine required gene-copy slots meeting
  their calibrated family-coverage thresholds; and those same nine functions in circular
  order with no excess qualifying copies. K and A\* are outside this synteny profile. DUST ≤0.9 is an online/checkpoint
  condition, not a repeated Arc final gate.
- **Set-level selection** clusters final passers at 99% identity and 95% coverage of both genomes and retains representatives. The
  rollout-relative diversity reward is therefore not interpreted as an intrinsic genome PASS/FAIL
  property.

This PhiX replication has no biological property deliberately targeted outside the viable lineage;
viable PhiX/Sinsheimervirus controls are therefore the default naturalness prior. A future directional
goal may leave its natural distribution, but its gate should be validated against known positives and
negatives while independent viability and safety gates remain. A proposed final gate can first be
replayed on saved per-candidate measurements; a new RL attempt is needed only if the change is adopted
in the online reward path.

### Sequence feasibility

| Objective (reward column)                             | Zero credit                                                                                                            | Full credit                                                                                                                                | Partial credit and rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `valid_nt_chars` (`reward_valid_nt_chars`)            | Any emitted character outside A/C/G/T.                                                                                 | No non-ACGT character is present.                                                                                                          | Binary. The raw helper regards an empty string as having no invalid character, but empty output fails length and aggregate nucleotide gates and cannot receive the safety-qualified GDPO objective. This prevents malformed sequence text from satisfying downstream biological tools.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `genome_length` (`reward_genome_length`)              | Length ≤3,000 nt or ≥5,800 nt.                                                                                         | 5,359–5,550 nt inclusive.                                                                                                                  | Linear lower taper over 3,000–5,359 nt and upper taper over 5,550–5,800 nt. Hard length screening is independently 5,306–5,730 nt. A 5,654-nt published viable example earns 0.584; a 5,730-nt insertion mutant earns 0.28. No-EOD caps at 6,016/6,024 nt earn zero length credit. Lower shaping remains anchored to the prior exact-FASTA Sinsheimervirus cohort (155 genomes, 5,339–5,388 nt; p5 5,359). The upper preference follows the instability observation above 5,550, not an assertion of inviability there; see the [1984 study](https://doi.org/10.1128/jvi.52.3.822-827.1984) and [King viable FASTA](https://github.com/ArcInstitute/evo2/blob/main/phage_gen/data/viable_generated_phage_genomes.fasta). PhiX has no terminal repeat; biological length includes prompt bases and excludes EOD/control tokens. |
| `gc_content` (`reward_gc_content`)                    | Mathematically at or below −5% or at or above 100%; only the 100% endpoint is physically attainable.                   | 30–65% inclusive.                                                                                                                          | Below 30%: `max(0, 1 - (30-GC)/35)`; above 65%: `max(0, 1 - (GC-65)/35)`. Thus 0% GC still scores 1/7, while 100% scores 0. The broad band rejects extreme composition while leaving room around the PhiX reference.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `nt_homopolymer` (`reward_nt_homopolymer`)            | No finite positive run length reaches exactly zero; an ineligible row is subsequently zeroed by the exact-safety mask. | Maximum nucleotide run *H* ≤ 10 bases.                                                                                                     | For *H* > 10, score `10/H`, decreasing asymptotically toward zero. Long homopolymers are discouraged because they are low-complexity and can complicate synthesis and sequencing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `dustmask_end` (`reward_dustmask_end`)                | No valid masked fraction reaches zero; an ineligible row is subsequently zeroed by the exact-safety mask.              | Maximum DUST-masked fraction *F* over either terminal 200-nt window ≤ 0.9.                                                                 | For 0.9 < *F* ≤ 1, score `0.9/F` (0.9–1.0). A failed external DUST command fails the scoring batch. The nucleotide-pass diagnostic supplies the binary cutoff; the final Arc target profile does not repeat this gate. DUST uses the approach described by [Morgulis et al.](https://doi.org/10.1089/cmb.2006.13.1028).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Nucleotide-pass diagnostic (`reward_nucleotide_pass`) | Any component gate fails.                                                                                              | All characters are A/C/G/T, length is 5,306–5,730 nt, GC is 30–65%, maximum homopolymer is ≤10, and both terminal DUST fractions are ≤0.9. | Binary acceptance and checkpoint diagnostic, not a separate GDPO shaping objective. Failure does not suppress independent ORF, protein, or synteny measurements; its non-DUST thresholds are also applied by final Arc QC.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

### Protein evidence, synteny, and diversity

The PHROGs consensus annotation search uses sensitivity 7.5 from
[`mmseqs_protein_database_sensitivity`](../configs/arc_genome_design_filtering_local.yaml).
It supplies required-function and synteny evidence for online rewards and final
screening. This recovers significant partial hits missed at sensitivity 4.0 while retaining
the existing E-value and native-coverage rules. Benchmark cost against the actual called
protein workload and thread allocation when adapting the recipe to larger genomes.

The [reference search](../src/bionemo/evo2_phage_gen/reward.py) uses MMseqs exhaustive
protein alignment at `-e 1` against all called candidate ORFs. It runs `createdb`,
`align`, and `convertalis` explicitly with a private NUL-terminated candidate list,
avoiding the unterminated `fake_pref` stream in the pinned `easy-search` workflow.
The candidate set and alignment settings are unchanged. Bypassing the heuristic prefilter prevents
otherwise significant homologs from disappearing before alignment after small
sequence changes. This search supplies graded evidence for synteny, tropism, and
gene-A origin; hard QC uses its separate searches and acceptance rules.

This search's target database is the current scoring call's ORFs, so E-values depend
on their total amino-acid count ([MMseqs implementation](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/alignment/EvalueComputation.h)).
Changing call size, genome lengths, or called protein content can change weak-hit scores
without changing the aligned pair. While the alignment stays fixed, increasing target
residues approximately multiplies E by the same factor. Strong hits retain full significance
credit while E remains ≤1e-5; weak hits can cross the E=1 search cutoff and disappear.
Synteny can move in either direction because weaker extra matches also reduce its copy penalty.
This applies to the shared reference rewards (synteny, tropism, and A evidence); the PHROGs
function search instead uses a fixed target database.

A September 14 offline replay of the same 96 generated genomes with reference-only
synteny (before the family-aware update) and the MMseqs revision linked above measured
the following absolute synteny-score changes. Pools
larger than 96 were filled with separately named copies; saved ORFs were held fixed.

| Scoring-call sizes compared | Mean absolute change | Largest absolute change |
| --------------------------- | -------------------: | ----------------------: |
| 1 vs 96                     |              0.00821 |                 0.02576 |
| 32 vs 96                    |              0.00060 |                 0.00338 |
| 96 vs 256                   |              0.00054 |                 0.00339 |
| 96 vs 768                   |              0.00078 |                 0.00780 |

The 32–768 pools retained the same hits for these generated genomes, and all comparisons
retained the same 12 full-length designs with synteny ≥0.9. Singleton searches admitted
213 additional weak hits. Small raw changes can still affect tightly grouped candidates:
96 vs 768 changed within-prompt standardized synteny by up to 1.39 in this validation bank.
These are fixed-bank observations, not a universal bound or a training-effect measurement.
The family-aware scorer can still use these direct-reference edges, but its combined
score has not been remeasured across these pool sizes; the table is historical evidence.
Keep the scoring-call context consistent for comparisons. Removing the continuous
significance factor alone leaves the search cutoff and cannot guarantee batch independence.

The shared [protein-match scorer](../src/bionemo/evo2_phage_gen/protein_evidence.py)
uses `w = (S * I * Q * T)**0.25`. Each factor is bounded on `[0, 1]`:

- `S` is significance: zero for E ≥ 1, one for E ≤ 1e-5 (including E = 0),
  and `-log10(E)/5` between those endpoints.
- `I = clip((p - 0.05)/(p_full - 0.05), 0, 1)`, where `p` is protein identity
  as a fraction and `p_full` is 0.90 for synteny/gene-A evidence or 0.95 for tropism.
- `Q` and `T` are native MMseqs query and target coverage, each divided by its
  full-credit target and capped at one: 0.95 for synteny or 0.99 for tropism.

Any zero factor gives zero match credit; positive factors receive continuous credit.
The four-term geometric mean has no additional raw-integrity cutoff, minimum-credit
bonus, or configurable exponent. The 5% identity baseline is a shaping choice inspired
by uniform amino-acid matching at fixed positions, not a significance threshold for
searched alignments. These are reference-match scores, not probabilities of protein
function. See the [sequence-alignment statistics discussion](https://www.ncbi.nlm.nih.gov/BLAST/tutorial/)
for the role of sequence composition and search selection in interpreting chance matches.

The identity targets above apply to **direct-reference** evidence. Synteny also uses
`score_function_matches`: every admitted PHROG-consensus hit in an allowed family
supplies `min(1,qcov/qmin,tcov/tmin)` credit without a 90% identity requirement.
Global assignment selects at most one ORF per function and one function per ORF;
a stronger unrelated-family hit cannot hide partial required-function evidence.
Each mapped slot uses the stronger route. Family recognition searches all consensuses, not
individual database members; the separate AAI search uses members. The
[function profile](../configs/required_genes.md#synteny-uses-the-same-function-definitions)
documents the mapping, short-J ORF/coverage calibration, and natural-genome replay.

Protein coverage uses MMseqs `qcov`/`tcov` (fraction of each protein spanned), not
`alnlen/length`: alignment columns can include gaps and inflate that estimate.
The raw alignment, query, and target lengths remain diagnostics. Older hit tables
without native coverage must be remeasured for coverage-dependent objectives; the
AAI definition remains mean best-hit percent identity, without a coverage filter.

These are PhiX case-study choices, not universal phage-viability requirements. In particular,
the optional AAI novelty criterion follows the [King et al. workflow](https://doi.org/10.1126/science.aec2657):
select the lowest-E-value hit against individual PHROGs proteins for each called ORF, then average
those percent identities. It is neither nucleotide identity to PhiX nor identity to synthetic
family consensuses; the annotation database remains separate. WT and known viable designs can
legitimately exceed the 95% novelty cutoff. Check both those controls and the published low-AAI
designs, rather than treating novelty rejection as evidence of nonviability.

Stage 00 automatically prepares this separate database with `--prepare-phrogs-member-database`.
The downloader verifies the recipe-pinned SHA256 of `FAA_phrog.tar.gz` and builds the individual
protein database selected by `mmseqs_db_aai_database`, which is required when AAI is enabled.
Changing the database changes the measurement and belongs in a new experiment.

| Objective (reward column)                                               | Zero credit                                                                                                                                                               | Full credit                                                                                                                                                                    | Partial credit and rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tropism` (`reward_external_tropism`)                                   | No called-ORF G match with E < 1, identity >5%, and positive coverage on both sides.                                                                                      | A G match with E ≤1e-5, identity ≥95%, and both coverages ≥0.99.                                                                                                               | Maximum four-term geometric-mean integrity among called-ORF matches to PhiX G, using the factors above. The independent hard gate requires identity ≥60% and both coverages ≥0.95. Partial evidence can guide RL before final QC passes. The G proxy follows the [PhiX design workflow](https://www.science.org/doi/10.1126/science.aec2657).                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `required_genes` (`reward_external_required_genes`)                     | Missing/invalid metrics or native coverage, an empty profile, or no allowed-family evidence.                                                                              | All configured named functions have distinct ORFs meeting both coverage targets.                                                                                               | Mean assigned coverage credit over the nine named A/B/C/D/E/F/G/H/J functions. Per-hit credit is `min(1,qcov/qmin,tcov/tmin)`; missing functions earn zero. Explicit PHROG alternatives fill one slot, including the viable alternate J; unrelated annotations and extra copies cannot replace missing functions. Coverage defaults to 0.75/0.75, with C 0.70/0.47, E 0.58/0.75, B 0.75/0.68, and alternate J (PHROG3780) 0.64/0.75. See the [biological rationale and limits](../configs/required_genes.md) for experimental evidence, K/A\* scope, and adaptation guidance.                                                                                                                                                                                   |
| `synteny` (`reward_external_synteny`)                                   | No positive match evidence, or the weighted synteny score is ≤0 after penalties.                                                                                          | All nine mapped function slots have full-credit one-to-one ORF matches in circular order, with no excess homolog mass. Supported family matches do not need 90% PhiX identity. | Each edge uses the stronger of direct PhiX integrity and allowed-family coverage credit. Maximum-weight one-to-one assignment gives content *C*; circular order-preserving assignment gives *O*; excess homolog mass gives *D*. Score `clip(0.25C + 0.75O - 0.75D, 0, 1)` with nine mapped slots. Final synteny uses the same families and coverage targets, requiring all nine functions in order with no extra qualifying copies. See [function-aware synteny](../configs/required_genes.md#synteny-uses-the-same-function-definitions).                                                                                                                                                                                                                      |
| `gene_a_origin` (`reward_gene_a_origin`)                                | No A match evidence, or no complete in-frame site in the accepted window with recognition, binding, and nicking match fractions all above 25%.                            | Full A integrity and one exact functional 28-nt site within ±30 nt of offset 345, in the same frame, with no extra strong sites.                                               | Score `(A × M × P × U)**0.25`, where P is position/frame eligibility and U is `1/max(1,strong_site_count)`, with baseline-adjusted motif score M defined below. The position/frame window is an acceptance gate within this reward; eligible offsets have no distance penalty. This is online shaping and a diagnostic, not a final hard gate.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `average_protein_identity` (`reward_external_average_protein_identity`) | No hit-bearing ORFs, missing AAI measurement, or missing output from an otherwise completed Arc run.                                                                      | Mean identity ≤95% with at least 10 hit-bearing ORFs.                                                                                                                          | Average the lowest-E-value individual PHROGs protein hit per called ORF, without a family-level reduction or consensus coverage gate. Score `novelty × min(hit_ORF_count/10,1)`: novelty is 1 through 95% and `max(0.25, (100-AAI)/5)` above 95%. Final Arc QC requires AAI ≤95% after its upstream gene-content gates; it does not require the reward's 10-ORF full-credit state. This optional divergence objective is separate from gene completeness and biological viability.                                                                                                                                                                                                                                                                              |
| `mmseqs_cluster_diversity` (`reward_mmseqs_cluster_diversity`)          | The genome fails the valid-character, hard-length, GC, or homopolymer prefilter, or is missing from MMseqs output. No finite cluster size otherwise reaches exactly zero. | A singleton across eligible prompts sharing the design goal in this scoring batch.                                                                                             | At 99% aligned nucleotide identity and ≥95% coverage of both genomes (`--cov-mode 0`, `--seq-id-mode 0`, `--cluster-mode 0`), a member of a cluster of size *N* scores `1/N`. Sequences retain their supplied start and strand; the default prompts share coordinate 1. Arbitrarily rotated near-clones are not guaranteed to share a cluster. A shared short gene alone cannot qualify. DUST is not part of this prefilter, and a failed MMseqs command fails the scoring batch. All prompts for the same design goal share this batch pool. This is within-batch diversity, not a per-genome viability gate; final passers use the same thresholds for separate set-level selection. [MMseqs2](https://doi.org/10.1038/nbt.3988) supplies the implementation. |

For gene-A origin, rescale each motif match fraction with `f(x) = max(0, (x - 0.25) / 0.75)`.
Let `r` cover the first 10 nt (recognition), `b` the next 18 nt (binding), and `n` positions
4–7 (the overlapping nicking core). The best eligible site supplies
`M = f(n)² × f(r) × f(b)`. The 25% anchor represents uniform-DNA matching for shaping;
it is not a significance threshold, and searching multiple offsets can give random windows
positive credit. Partial sites receive credit without meeting the strong-site criteria.
The genome-wide strong-site count uses raw `r ≥ 8/10`, `b ≥ 14/18`, and an exact nicking
core solely for the uniqueness factor. All four factors enter the geometric mean; two
strong sites therefore multiply the reward by `0.5**0.25 ≈ 0.841`. Any zero factor still
gives zero reward. Positions 29–30 are ignored because the cited
[cleavage experiments](https://www.sciencedirect.com/science/article/pii/S0021925820820791)
found them dispensable. See [`score_gene_a_origin`](../src/bionemo/evo2_phage_gen/protein_evidence.py).

Identity and coverage are separate constraints. The 99%/95% near-duplicate rule allows modest
end edits to remain in the same cluster: lowering coverage from 98% to 95% makes those edits
less likely to earn novelty credit. It is not an indel penalty; edits exceeding the coverage
allowance can still separate an otherwise identical backbone, so length and gene-integrity
objectives still matter. This is an intentional departure
from ARC's 99%-identity clustering with inherited 80% coverage, not a reinterpretation of past runs.

### Mandatory whole-genome safety objectives

All three classes are required for the PhiX bacterial-host profile. Each class uses the same
categorical mapping: `PASS → 1`, a review-eligible measured `INDETERMINATE` with findings `→ 0.25`,
and `FAIL`, unmeasured, invalid, or tool-failed `→ 0`. The hard safety gate still requires all
required classes to be `PASS`; partial credit never qualifies a candidate for acceptance. This
separation follows the conservative screening posture in `configs/phage_safety_policy.yaml` and
regulatory emphasis on excluding detrimental genes and temperate behavior in [EMA's phage-therapy
quality guideline](https://www.ema.europa.eu/en/quality-aspects-phage-therapy-medicinal-products).

| Objective (reward column)                    | Full-credit state | What the class screens                                                                   |
| -------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `safety_amr` (`reward_safety_amr`)           | `PASS`            | Acquired antimicrobial-resistance determinants.                                          |
| `safety_toxin` (`reward_safety_toxin`)       | `PASS`            | Toxin and virulence-associated protein evidence.                                         |
| `safety_lysogeny` (`reward_safety_lysogeny`) | `PASS`            | Integrases, repressors, excision machinery, and other lysogeny/temperate-phage evidence. |
