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
| Biological representatives after circular/RC deduplication |       1,000 |
| Submitted to safety / excluded by pre-safety QC            |     991 / 9 |
| Safety PASS / FAIL / INDETERMINATE                         | 989 / 0 / 2 |
| Safety-PASS target hard-QC representatives                 |         513 |
| Post-QC 99%-identity clusters and accepted representatives |         511 |

The denominators proceed from raw generation through biological deduplication, safety screening,
and target hard QC, then post-QC clustering. Likelihood is a within-protocol ranking signal; these
computational candidates are not evidence of bootability or wet-lab safety. See the
[case-study notes](../skills/bionemo-phage-design/references/case-study-results.md) for historical context.

That completed run used the earlier single-origin prompt semantics. The current launcher starts a
new `results/phix174-8xh100-mixed-anchors` result root and does not resume that run in place.

## Quick start

Run from `recipes/evo2_phage_gen`. The PhiX follow-up uses the published-lineage `7b-base` checkpoint
with the current mixed-anchor defaults. Supplying `--sampling-selection` explicitly skips the
fresh-calibration review stop:

```bash
./.ci_build.sh
tmux new -s phix174-e2e
./examples/phix174_8xh100.sh \
  --model-variant 7b-base \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
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
./examples/phix174_8xh100.sh --prepare-only --result-root "$PWD/results/phix174-8xh100-mixed-anchors"

# Resume the RL stage of the 7b-base run.
./examples/phix174_8xh100.sh --resume-from 40 --model-variant 7b-base --result-root "$PWD/results/phix174-8xh100-mixed-anchors"

# Use an explicitly reviewed sampling selection, without blocking if the automatically identified
#  top setting differs. In practice automatic selection can be noisy, and these settings should work
#  well for phix174.
./examples/phix174_8xh100.sh --sampling-selection examples/default-sampling-selection.yaml --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
```

Completed stages and substages are skipped. Reuse the same result root and sampling selection when
resuming; `--resume-from` selects where checking resumes but does not create missing state or
completion markers. A material prompt, reward, sampling, or model change should start a new result
root. Stage 20 records its real-data restart smoke separately as `stages/20-sft-smoke.done`; it
validates and reuses a complete converted base checkpoint instead of redownloading or reconverting
it. Stage 40 likewise keeps `stages/40-pilot.done`, `stages/40-pilot-reload.done`,
`stages/40-pilot-check.done`, and `stages/40-rl.done` distinct, and reuses only a validated schema-2
optimizer-free prepared SFT checkpoint. Do not create a marker unless its operation is known to
have succeeded.

### Optional W&B logging

Authenticate with `wandb login` (or provide `WANDB_API_KEY` through a secret manager), then opt in:

```bash
./examples/phix174_8xh100.sh --wandb --wandb-entity YOUR_ENTITY \
  --wandb-sft-project evo2-phage-design-sft \
  --wandb-rl-project evo2-phage-design-gdpo \
  --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
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
does not substitute for the deployed full-shape pilot. For the circular
PhiX reference, the deployed
16- and 24-nt prompts mix four 1-based anchors: coordinate 1 (`origin`), 2,387 (`before_g`),
3,918 (`after_h`), and 3,973 (`a_cluster_start`). `before_g` and `a_cluster_start` begin eight
bases before the curated G and A starts, respectively, so they preserve each start codon and the
first 8 or 16 coding bases while leaving the rest of G, A, A\*, and B generative. `after_h` supplies
an unseeded route into the A/A\*/B cluster; `origin` retains the canonical context while fixing a
short origin-spanning segment shared by those three ORFs. This rotation strategy does not apply to
a linear genome, whose biological endpoints must be preserved.

Each 768-rollout GDPO update uses 16 prompt records × 48 generations. With DP8, each 96-request
decode batch contains two prompt records, while the global update contains all eight anchor×length
strata: two prompt records and 96 generated sequences per stratum. Validation remains a separate
96-record mixture. The 1,000-design final rollout uses 125 prompts per stratum: two 500-record,
same-length files alternate four anchors and combine to exactly 1,000 records.
Each GPU generates its shard in packed batches of 96 by default; set
`FINAL_PROMPT_BATCH_SIZE` only when qualifying a different device profile.
The previous 5,420-token/5,632-context 768-point setting completed three 8×H100 rollout/QC/replay/update
cycles with zero replay masks, including validation after update two, a post-validation update, an optimizer-inclusive checkpoint,
and a fresh-process exact reload. Warm update two took 492.65 seconds for generation including
233.78 seconds of scoring, 37.74 seconds for policy/reference replay, and 54.60 seconds for the
policy update. A 1,024-point attempt failed recurrent-state allocation before decode, so it is not
a supported one-wave setting.
Legacy synchronous GRPO is bounded by both steps and epochs; this 96-row bank supplies six updates
per epoch, so the configured 500 epochs safely exceeds the requested 500-step ceiling.

The launcher sets `max_model_len` to the smallest 256-token boundary covering the longest selected
prompt, its two-token `+~` prefix, and `max_new_tokens` (6,144 with the defaults), rather than
allocating the unused remainder of the 10,240-token SFT context. Native RL trajectories retain the
first sampled EOD and its log-probability, mask only synthetic padding, and exclude EOD and any
post-EOD physical samples from biological QC. Filtered policy replay also keeps each sampled action
in a normalized target-preserving support; generation-versus-replay error telemetry remains enabled.
Qualification reports authentic EOD, capped-without-EOD, and below-cap-without-EOD as three
exclusive outcomes. The `phage_qc/termination/*` scalars retain counts and rates for those outcomes
and place authentic-EOD genomes into below-lower-zero, lower-taper, full-credit, upper-taper, and
at-or-above-upper-zero bins, so fixed-bank placement direction remains recoverable without raw rows.

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
./examples/phix174_8xh100.sh --calibrate-only --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
```

Inspect `results/phix174-8xh100-mixed-anchors/calibration/scoring/selection-evidence.csv` and its neighboring
score/novelty artifacts. Prefer eligible settings with working metrics, useful hard-QC signal,
low copying, diverse outputs, and a stable quality-diversity plateau. An agent may perform this
review and write the custom choice when the user delegates it. Copy the example YAML, including its
named `prompt_anchors`, then continue without repeating the completed sweep:

```bash
cp examples/default-sampling-selection.yaml /tmp/phix174-sampling.yaml
# Edit /tmp/phix174-sampling.yaml, then:
./examples/phix174_8xh100.sh --resume-from 30 --sampling-selection /tmp/phix174-sampling.yaml --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
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

Before GDPO, the exact configured environment scores the coordinate origin and both deployed
reference rotations together and requires identical reward, filter, and measurement-support
outcomes. Arc hard QC removes ORFipy calls beginning wholly inside its appended pseudocircular tail,
while retaining cross-origin ORFs, so tail length cannot create rotation-dependent duplicate genes.
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

This is the human-readable contract for the 15 objectives in
`configs/gdpo_phage_megatron.yaml`. It is also the worked example for the run-specific
`artifacts/RL_SCORE_DEFINITIONS.md` that an agent writes when designing or changing objectives;
the E2E shell script does not generate that artifact. These thresholds reproduce the current
PhiX174 computational profile, not universal phage-design optima or evidence of bootability.

GDPO receives each objective row below as a separate `[0, 1]` objective. The scalar `weight_*`
settings are diagnostic and do not reweight objectives after GDPO normalization. The first 12
objectives are forced to zero unless the sequence has an exact sequence-safety `PASS`; the three
safety objectives remain unmasked so failures still provide learning signal. Missing, invalid, non-finite,
or unavailable measurements map to zero when scoring returns a row. Configured Arc, DUST, or
diversity-command failures can instead fail the scoring batch rather than fabricate a biological
score. Final checkpoint selection uses the stricter safety-qualified, full-QC,
cluster-deduplicated pass rate rather than mean reward.

### Objective implementation map

The ordered objective inventory and reward-column wiring live in the
[GDPO configuration](../configs/gdpo_phage_megatron.yaml). The environment validates and
assembles that matrix in
[`gdpo_objective_scores_from_scored`](../src/bionemo/evo2_phage_gen/nemo_rl_env.py), including
the exact-safety mask. The implementations for the individual terms are:

| Objective                  | Primary implementation                                                                                                                                                                                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `valid_nt_chars`           | [`has_valid_nt_chars`](../src/bionemo/evo2_phage_gen/qc.py) and [`score_nucleotide_metrics`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                             |
| `genome_length`            | [`_genome_length_score`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                                                                                 |
| `gc_content`               | [`calculate_gc_content`](../src/bionemo/evo2_phage_gen/qc.py) and [`_interval_score`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                    |
| `nt_homopolymer`           | [`calculate_nt_homopolymer_len`](../src/bionemo/evo2_phage_gen/qc.py) and [`_upper_bound_ratio_score`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                   |
| `dustmask_end`             | [`calculate_dustmasker_metrics`](../src/bionemo/evo2_phage_gen/qc.py) and [`score_nucleotide_metrics`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                   |
| `protein_hit_count`        | [`_add_mmseqs_hit_rewards`](../src/bionemo/evo2_phage_gen/reward.py) and [`add_protein_alignment_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py)                                                                                                    |
| `tropism`                  | [`smooth_protein_match_integrity`](../src/bionemo/evo2_phage_gen/protein_evidence.py), [`summarize_smooth_reference_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py), and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py) |
| `required_genes`           | [`summarize_required_gene_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_required_gene_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                               |
| `synteny`                  | [`smooth_protein_match_integrity`](../src/bionemo/evo2_phage_gen/protein_evidence.py), [`score_smooth_reference_architecture`](../src/bionemo/evo2_phage_gen/protein_evidence.py), and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py) |
| `gene_a_origin`            | [`score_gene_a_origin`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                         |
| `average_protein_identity` | [`summarize_best_hit_aai`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_average_protein_identity_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                              |
| `mmseqs_cluster_diversity` | [`add_mmseqs_cluster_diversity_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                                                                 |
| `safety_amr`               | [`run_amrfinder_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                 |
| `safety_toxin`             | [`run_toxin_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                     |
| `safety_lysogeny`          | [`run_phrogs_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                    |

### How rewards, gates, and selection differ

- **RL objectives** are the rows below. Their `[0, 1]` values shape training; full credit is a target,
  not an automatic final acceptance boundary.
- **Online measurement** disables Arc's length prefilter so a tool-safe length outlier still receives
  independent ORF, protein, and architecture measurements. Length remains its own graded reward and
  final acceptance gate.
- **Nucleotide-pass telemetry** records the binary conjunction used by checkpoint and final-QC
  diagnostics; it is not a separate GDPO objective duplicating the graded component terms.
- **Checkpoint retention and selection** keep complementary evidence. NeMo-RL's managed top three
  use fixed-bank `mean_reward` (plus its latest resumable checkpoint), while the launcher hard-links
  the best aggregate checkpoint and the best positive
  `binary_safety_qualified_full_qc_cluster_deduplicated_rate` checkpoint independently. Final
  selection prefers a strict-positive checkpoint, breaking ties by aggregate reward and then step;
  if none exists, it selects the best non-boundary aggregate checkpoint and records
  `strict_endpoint_qualified: false`. The strict endpoint requires exact safety `PASS`, full credit
  on the binary-core rewards—including the 5,359–5,550-nt length band—then the independent external
  hard-pass flags, and counts one representative per online 99%-identity/95%-coverage cluster. Smooth synteny,
  tropism, A-origin, required-gene, and AAI targets are shaping terms rather than implicit hard
  gates; aggregate fallback is not a hard-QC pass.
- **Final per-genome QC** uses exact safety `PASS` plus the Arc target-profile waterfall: A/C/G/T
  only; length 5,306–5,730 nt; GC 30–65%; homopolymer ≤10; at least seven distinct PHROG families
  with ≥0.75 query and target coverage; a PhiX G hit at 60–100% identity with ≥0.95 query and target
  coverage; mean per-ORF PHROGs member identity 0–95%; all nine required gene-copy slots meeting
  their control-calibrated coverage thresholds; and one-to-one circular synteny with at most one
  missing reference locus and no excess homolog copies. DUST ≤0.9 is an online/checkpoint
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
| Nucleotide-pass diagnostic (`reward_nucleotide_pass`) | Any component gate fails.                                                                                              | All characters are A/C/G/T, length is 5,306–5,730 nt, GC is 30–65%, maximum homopolymer is ≤10, and both terminal DUST fractions are ≤0.9. | Binary acceptance and checkpoint diagnostic, not a separate GDPO shaping objective. Failure does not suppress independent ORF, protein, or architecture measurements; its non-DUST thresholds are also applied by final Arc QC.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

### Protein evidence, architecture, and diversity

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
protein database selected by `mmseqs_db_aai_database`. Existing configs without that key retain
the legacy AAI path; changing the database changes the measurement and belongs in a new experiment.

| Objective (reward column)                                               | Zero credit                                                                                                                                                               | Full credit                                                                                                                                                | Partial credit and rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `protein_hit_count` (`reward_external_protein_hit_count`)               | No measured PHROGs hit, missing alignment lengths, or missing output from an otherwise completed Arc run.                                                                 | Effective unique-family coverage ≥7; the hard gate separately requires at least 7 unique families meeting the calibrated coverage threshold.               | For each PHROGs target family, retain the best `min(query_coverage,target_coverage)` and sum across unique families; reward `min(sum/7,1)`. Presence credit is identity-independent. The PHROG-consensus threshold is 0.75, calibrated because consensus lengths differ systematically from member proteins; the exact PhiX spike gate remains 0.95. Duplicate ORFs cannot inflate family evidence, and fragments cannot pass the hard gate.                                                                                                                                                                                 |
| `tropism` (`reward_external_tropism`)                                   | No significant called-ORF match to PhiX G, or raw integrity ≤0.001.                                                                                                       | E-value ≤1e-5, identity ≥95%, and both reference and candidate-ORF coverage ≥0.99.                                                                         | For E-values from 1 to 1e-5, significance ramps linearly in `-log10(E)`; multiply it by `(min(identity/0.95,1) × min(reference_coverage/0.99,1) × min(candidate_coverage/0.99,1))^1.5`, then rescale values above the 0.001 shuffled-null floor to start at 0.01. The independent hard gate remains identity ≥60% and both coverages ≥0.95, so low or partial evidence can guide RL without passing final QC. The G proxy follows the [PhiX design workflow](https://www.science.org/doi/10.1126/science.aec2657).                                                                                                           |
| `required_genes` (`reward_external_required_genes`)                     | Missing metrics or native coverage, an empty required-label definition, or zero evidence.                                                                                 | All nine required gene-copy slots meet their calibrated query/target coverage targets.                                                                     | Score `(coverage_sum/total) × min(total/9,1)` using per-edge coverage normalized to the configured family targets. The default is 0.75/0.75; C uses 0.70/0.47, E 0.58/0.75, and B 0.75/0.68 based on observed viable designs. C's native target coverage is 32/68, not the gap-inflated 33/68 alignment-column ratio. One-to-one assignment prevents an ORF or PHROG target from satisfying both B/D head-morphogenesis slots. K is optional; the 75-nt ORF minimum admits alternate J. A\* remains outside this ORF-caller-based term. These are PhiX control-calibrated checks, not universal essential-gene requirements. |
| `synteny` (`reward_external_synteny`)                                   | No significant called-ORF match exceeds the 0.001 shuffled-null integrity floor.                                                                                          | All callable reference loci have one-to-one ORF matches at ≥90% identity and ≥0.95 coverage on both sides, in circular order, with no excess homolog mass. | Each edge uses the same significance ramp and convex integrity formula as tropism, with 90% identity and 0.95 reciprocal-coverage full-credit targets. Maximum-weight one-to-one assignment gives content *C*; circular order-preserving assignment gives *O*; excess homolog mass gives *D*. Score `clip(0.25C + 0.75O - 0.75D, 0, 1)` with a fixed reference-locus denominator. Properly ordered weak homologs outscore the same scrambled homologs, while deletion, fusion, and duplication cannot improve the score. Hard LoVis clustering remains a separate final-pass measurement.                                    |
| `gene_a_origin` (`reward_gene_a_origin`)                                | No assigned A ORF with a credible functional origin near its expected in-frame offset, or no A match evidence.                                                            | A has full match integrity and one exact functional 28-nt site within ±30 nt of offset 345, in the same frame.                                             | Score the 10-nt recognition region, positions 4–7 around the nick quadratically, and the following 18-nt binding region; multiply by A-match integrity and `1/strong_site_count`. The accepted offset window no longer penalizes viable in-frame shifts. Positions 29–30 of the 30-nt reference motif do not affect this score because [cleavage experiments](https://www.sciencedirect.com/science/article/pii/S0021925820820791) found them dispensable. This is online shaping and a diagnostic, not a final hard gate.                                                                                                   |
| `average_protein_identity` (`reward_external_average_protein_identity`) | No hit-bearing ORFs, missing AAI measurement, or missing output from an otherwise completed Arc run.                                                                      | Mean identity ≤95% with at least 10 hit-bearing ORFs.                                                                                                      | Average the lowest-E-value individual PHROGs protein hit per called ORF, without a family-level reduction or consensus coverage gate. Score `novelty × min(hit_ORF_count/10,1)`: novelty is 1 through 95% and `max(0.25, (100-AAI)/5)` above 95%. Final Arc QC requires AAI ≤95% after its upstream gene-content gates; it does not require the reward's 10-ORF full-credit state. This optional divergence objective is separate from gene completeness and biological viability.                                                                                                                                           |
| `mmseqs_cluster_diversity` (`reward_mmseqs_cluster_diversity`)          | The genome fails the valid-character, hard-length, GC, or homopolymer prefilter, or is missing from MMseqs output. No finite cluster size otherwise reaches exactly zero. | A singleton within its prompt group.                                                                                                                       | At 99% aligned nucleotide identity and ≥95% coverage of both genomes (`--cov-mode 0`, `--seq-id-mode 0`, `--cluster-mode 0`), a member of a cluster of size *N* scores `1/N`. Circular inputs are canonicalized. A shared short gene alone cannot qualify. DUST is not part of this prefilter, and a failed MMseqs command fails the scoring batch. This is within-batch diversity, not a per-genome viability gate; final passers use the same thresholds for separate set-level selection. [MMseqs2](https://doi.org/10.1038/nbt.3988) supplies the implementation.                                                        |

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
