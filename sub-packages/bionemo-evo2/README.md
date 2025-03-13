# bionemo-evo2

`bionemo-evo2` is a `pip`-installable package that contains **data preprocessing**, **training**, and **inferencing** code for Evo2, a new `Hyena`-based foundation model for genome generation and understanding. Built upon `Megatron-LM` parallelism and `NeMo2` algorithms, `bionemo-evo2` provides the remaining tools necessary to effectively fine-tune the pre-trained Evo2 model checkpoint on user-provided sequences at scale, and generate state-of-the-art life-like DNA sequences from Evo2 for downstream metagenomic tasks.

## Quickstart tutorials

Please see

## Installation

To install this package, execute the following command:

```bash
pip install -e .
```

To run unit tests, execute the following command:

```bash
pytest -v .
```

## Preprocessing

To train or fine-tune Evo2 on a custom dataset, we need to preprocess and index sequence data for training from raw FASTA files into tokenized binaries compliant with `NeMo2` / `Megatron-LM`. For more information about how to configure your data for training, refer to [data/README.md](src/bionemo/evo2/data/README.md) and [utils.config.Evo2PreprocessingConfig](src/bionemo/evo2/utils/config.py).

```bash
preprocess_evo2 -c <CONFIG_PATH>
```

## Training

Given a preprocessed collection of preprocessed datasets, and optionally a pre-trained NeMo2 checkpoint for Evo2, training can be executed using the following command:

```bash
$ train_evo2 --help
usage: train_evo2 [-h] (-d DATASET_CONFIG | --mock-data) [--dataset-dir DATASET_DIR] [--num-nodes NUM_NODES] [--devices DEVICES] [--seq-length SEQ_LENGTH]
                  [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE] [--context-parallel-size CONTEXT_PARALLEL_SIZE]
                  [--wandb-project WANDB_PROJECT] [--wandb-run-id WANDB_RUN_ID] [--wandb-group WANDB_GROUP] [--wandb-job-type WANDB_JOB_TYPE] [--wandb-offline]
                  [--wandb-anonymous] [--create-tensorboard-logger] [--create-tflops-callback] [--max-steps MAX_STEPS]
                  [--sequence-parallel] [--fp8] [--micro-batch-size MICRO_BATCH_SIZE] [--global-batch-size GLOBAL_BATCH_SIZE]
                  [--grad-acc-batches GRAD_ACC_BATCHES] [--val-check-interval VAL_CHECK_INTERVAL] [--grad-reduce-in-fp32] [--fp8-wgrad]
                  [--use-megatron-comm-overlap-llama3-8k] [--tp-comm-overlap-backend {nccl,mpi,gloo}] [--align-param-gather] [--add-bias-output]
                  [--model-size {1b,1b_nv,40b,40b_arc_longcontext,40b_nv,7b,7b_arc_longcontext,7b_nv,test,test_nv}] [--result-dir RESULT_DIR] [--experiment-name EXPERIMENT_NAME]
                  [--limit-val-batches LIMIT_VAL_BATCHES] [--log-every-n-steps LOG_EVERY_N_STEPS] [--ckpt-dir CKPT_DIR] [--wd WD] [--restore-optimizer-from-ckpt]
                  [--no-average-in-collective] [--seed SEED] [--workers WORKERS] [--gc-interval GC_INTERVAL] [--enable-preemption] [--ckpt-async-save]
                  [--ckpt-format {torch_dist,zarr}] [--eod-pad-in-loss-mask] [--cross-entropy-loss-fusion] [--no-fp32-residual-connection]
                  [--debug-ddp-parity-freq DEBUG_DDP_PARITY_FREQ] [--hybrid-override-pattern HYBRID_OVERRIDE_PATTERN] [--num-layers NUM_LAYERS]
                  [--log-parameters-and-shapes] [--lr LR] [--min-lr MIN_LR] [--warmup-steps WARMUP_STEPS] [--nsys-profiling] [--nsys-start-step NSYS_START_STEP]
                  [--nsys-end-step NSYS_END_STEP] [--no-renormalize-loss] [--nsys-ranks NSYS_RANKS [NSYS_RANKS ...]]
                  [--activation-checkpoint-recompute-num-layers ACTIVATION_CHECKPOINT_RECOMPUTE_NUM_LAYERS] [--disable-checkpointing] [--clip-grad CLIP_GRAD]
                  [--seq-len-interpolation-factor SEQ_LEN_INTERPOLATION_FACTOR] [--overlap-param-gather] [--overlap-grad-reduce] [--hidden-dropout HIDDEN_DROPOUT]
                  [--attention-dropout ATTENTION_DROPOUT] [--no-activation-checkpointing | --selective-activation-checkpointing]

Train a Hyena model using NeMo 2.0.

options:
  -h, --help            show this help message and exit
  -d DATASET_CONFIG, --dataset-config DATASET_CONFIG
                        Path to the blended / weighted training dataset configuration YAML. (default: None)
  --mock-data           Train with Mock data (for testing/debugging), either set this or provide a dataset config. (default: False)
  --dataset-dir DATASET_DIR
                        Absolute path to the dataset directory. Defaults to using the absolute or relative paths (dataset_prefix) specified in the dataset config YAML.
                        (default: None)
  --num-nodes NUM_NODES
                        Number of nodes to use for training, defaults to 1. (default: 1)
  --devices DEVICES     Number of devices to use for training, defaults to 1. (default: 1)
  --seq-length SEQ_LENGTH
                        Training sequence length (default: 8192)
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Order of tensor parallelism. Defaults to 1. (default: 1)
  --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE
                        Order of pipeline parallelism. Defaults to 1. (default: 1)
  --context-parallel-size CONTEXT_PARALLEL_SIZE
                        Order of context parallelism. Defaults to 1. (default: 1)
  --create-tensorboard-logger
                        Create a TensorBoard logger. (default: False)
  --create-tflops-callback
                        Create a TFlops callback to log model FLOPs per training step. (default: False)
  --wandb-project WANDB_PROJECT
                        Wandb project name (default: bionemo_evo2)
  --wandb-run-id WANDB_RUN_ID
                        Wandb run identifier (default: None)
  --wandb-group WANDB_GROUP
                        A unique string shared by all runs in a given group (default: None)
  --wandb-job-type WANDB_JOB_TYPE
                        A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group. (default: None)
  --wandb-offline       Use wandb in offline mode (default: False)
  --wandb-anonymous     Enable or explicitly disable anonymous logging (default: False)
  --sequence-parallel   Set to enable sequence parallelism. (default: False)
  --fp8                 Set to enable FP8 (default: False)
  --micro-batch-size MICRO_BATCH_SIZE
                        Micro-batch size for data-parallel training. (default: 1)
  --global-batch-size GLOBAL_BATCH_SIZE
                        Global batch size for training. If set to None, infer it from the TP, CP, and PP parameters. (default: None)
  --grad-acc-batches GRAD_ACC_BATCHES
                        Number of batches to accumulate gradients over. (default: 1)
  --max-steps MAX_STEPS
                        Number of training optimizer update steps. (default: None)
  --val-check-interval VAL_CHECK_INTERVAL
                        Number of steps between validation measurements and model checkpoints. (default: None)
  --grad-reduce-in-fp32
                        Gradient reduce in FP32. (default: False)
  --fp8-wgrad           Faster option that is maybe less accurate (TBD) when using fp8. (default: False)
  --use-megatron-comm-overlap-llama3-8k
  --tp-comm-overlap-backend {nccl,mpi,gloo}
                        TP communication backend to use. Defaults to 'nccl'. (default: nccl)
  --align-param-gather
  --model-size {1b,1b_nv,40b,40b_arc_longcontext,40b_nv,7b,7b_arc_longcontext,7b_nv,test,test_nv}
                        Model architecture to use, choose between 7b, 40b, or test (a sub-model of 4 layers, less than 1B parameters). '_arc_1m' models have GLU / FFN
                        dimensions that support 1M context length when trained with TP<=8. (default: 7b)
  --add-bias-output     Add bias to the output layer to enable learning a simple prior. (default: False)
  --result-dir RESULT_DIR
                        Directory to write model checkpoints and results to. (default: None)
  --experiment-name EXPERIMENT_NAME
                        Directory to write model checkpoints and loggers outputs is RESULT_DIR/EXPERIMENT_NAME. (default: None)
  --limit-val-batches LIMIT_VAL_BATCHES
                        Number of validation steps (default: 20)
  --log-every-n-steps LOG_EVERY_N_STEPS
                        Number of steps between logging. (default: 1)
  --ckpt-dir CKPT_DIR   Directory to restore an initial checkpoint from. Use this for supervised fine-tuning. (default: None)
  --wd WD               Weight decay for optimizer. (default: 0.01)
  --restore-optimizer-from-ckpt
                        Restore optimizer state from initial checkpoint. Defaults to False. (default: False)
  --no-average-in-collective
                        Avaerage optimizer state in collective rather than dividing by dp size and summing. (default: False)
  --seed SEED           Set random seed for training. (default: 1234)
  --workers WORKERS     Number of workers to use for data loading. (default: 8)
  --gc-interval GC_INTERVAL
                        Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps. (default: 0)
  --enable-preemption   Enable preemption hooks. If enabled this will save a checkpoint whenver slurm exits. (default: False)
  --ckpt-async-save
  --ckpt-format {torch_dist,zarr}
                        Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated. Only use if resuming training from a zarr checkpoint. (default:
                        torch_dist)
  --eod-pad-in-loss-mask
                        Do not predict EOD/Pad tokens (typical default, but not default in original evo2). (default: False)
  --cross-entropy-loss-fusion
                        Use the faster, but maybe less accurate fused form of cross entropy, which also has bf16 grads internally. (default: False)
  --no-fp32-residual-connection
                        If set, turn off fp32 residual connections which may be faster but may impact accuracy. (default: False)
  --debug-ddp-parity-freq DEBUG_DDP_PARITY_FREQ
                        Set to value > 0 to debug DDP weight parity between ranks. (default: 0)
  --hybrid-override-pattern HYBRID_OVERRIDE_PATTERN
                        Override the hybrid override pattern in the config (specifies hyena layer ordering and type). (default: None)
  --num-layers NUM_LAYERS
                        If set, override the number of layers specified in the requested config. (default: None)
  --tflops-callback     Enable tflops calculation callback for Hyena / Evo2. Defaults to False. (default: False)
  --log-parameters-and-shapes
                        Log training parameters shapes and dtypes for debugging. (default: False)
  --lr LR               Learning rate. (default: 0.0003)
  --min-lr MIN_LR       Min learning rate in cosine annealing. (default: 3e-05)
  --warmup-steps WARMUP_STEPS
                        Number of warmup steps in cosine annealing (default: 2500)
  --nsys-profiling      Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with
                        `nsys`. For example: `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-
                        end=stop [regular python command here]` (default: False)
  --nsys-start-step NSYS_START_STEP
                        Start nsys profiling after this step. (default: 0)
  --nsys-end-step NSYS_END_STEP
                        End nsys profiling after this step. (default: None)
  --no-renormalize-loss
                        Do not renormalize the loss weights. (default: False)
  --nsys-ranks NSYS_RANKS [NSYS_RANKS ...]
                        Enable nsys profiling for these ranks. (default: [0])
  --activation-checkpoint-recompute-num-layers ACTIVATION_CHECKPOINT_RECOMPUTE_NUM_LAYERS
                        If set, override the default value set in the config. (default: None)
  --disable-checkpointing
                        Disable creating a ModelCheckpoint callback. (default: True)
  --clip-grad CLIP_GRAD
                        Grad clip value. Note that when using DDP this may need to be inflated. (default: 1.0)
  --seq-len-interpolation-factor SEQ_LEN_INTERPOLATION_FACTOR
                        Adjusts the linear scaling of ROPE (Rotary Position Embedding) for context extension. Set this factor relative to your base context length e.g., for
                        an original context length of 8192 and an extended context length of 524288, use 524288/8192 = 64. (default: None)
  --overlap-param-gather
                        Overlap the parameter gather with the optimizer step. This is currently disabled due to a NeMo bug when using DDP. Making this an option defaulting to
                        False is a temporary solution until the bug is fixed. (default: False)
  --overlap-grad-reduce
                        Overlap the gradient reduce with the optimizer step. (default: False)
  --hidden-dropout HIDDEN_DROPOUT
                        Dropout probability for the hyena layers (default: 0.0)
  --attention-dropout ATTENTION_DROPOUT
                        Dropout probability for the attention layers. (default: 0.0)
  --no-activation-checkpointing
  --selective-activation-checkpointing
```

To supply a pre-trained checkpoint, pass the NeMo2 checkpoint directory to `--ckpt-dir`, and the script will dump newly trained checkpoints and logs to `--experiment-dir`. However, if there are existing well-defined checkpoints in the directory specified by `--experiment-dir`, the script will automatically resume training from the most recent checkpoint in the experiment directory instead of starting from the checkpoint specified by `--ckpt-dir`, which streamlines long training sessions. (To disable this behavior, supply a new or clean `--experiment-dir` when restarting from `--ckpt-dir`.)

Training data and sampling weights can be specified using the `--dataset-config` argument as a YAML file adhering to the following schema: [utils.config.Evo2BlendedDatasetConfig](src/bionemo/evo2/utils/config.py). For more information about dataset sampling and blending during training with Megatron-LM, refer to [megatron/core/datasets/readme.md](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/readme.md). For example:

```yaml
- dataset_prefix: /workspace/bionemo2/data/metagenomics/pretraining_data_metagenomics/data_metagenomics_train_text_CharLevelTokenizer_document
  dataset_split: train
  dataset_weight: 0.18
- dataset_prefix: /workspace/bionemo2/data/gtdb_imgpr/pretraining_data_gtdb_imgpr/data_gtdb_imgpr_train_text_CharLevelTokenizer_document
  dataset_split: train
  dataset_weight: 0.24
- dataset_prefix: /workspace/bionemo2/data/imgvr_untagged/imgvr_untagged_data/data_imgvr_train_text_CharLevelTokenizer_document
  dataset_split: train
  dataset_weight: 0.03
- dataset_prefix: /workspace/bionemo2/data/promoters/pretraining_data_promoters/data_promoters_valid_text_CharLevelTokenizer_document
  dataset_split: validation
  dataset_weight: 0.0003
- dataset_prefix: /workspace/bionemo2/data/organelle/pretraining_data_organelle/data_organelle_valid_text_CharLevelTokenizer_document
  dataset_split: validation
  dataset_weight: 0.005
- dataset_prefix: /workspace/bionemo2/data/metagenomics/pretraining_data_metagenomics/data_metagenomics_test_text_CharLevelTokenizer_document
  dataset_split: test
  dataset_weight: 0.18
- dataset_prefix: /workspace/bionemo2/data/gtdb_v220/gtdb_v220_imgpr_merged_data/data_gtdb_imgpr_test_text_CharLevelTokenizer_document
  dataset_split: test
  dataset_weight: 0.24
```

## Inference

Once you have a pre-trained or fine-tuned Evo2 checkpoint, you can also prompt the model to generate DNA sequences using the following command:

```bash
$ infer_evo2 --help
usage: infer_evo2 [-h] [--prompt PROMPT] --ckpt-dir CKPT_DIR [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P] [--max-new-tokens MAX_NEW_TOKENS] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE] [--context-parallel-size CONTEXT_PARALLEL_SIZE] [--output-file OUTPUT_FILE]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       Prompt to generate text from Evo2. Defaults to a phylogenetic lineage tag for E coli.
  --ckpt-dir CKPT_DIR   Path to checkpoint directory containing pre-trained Evo2 model.
  --temperature TEMPERATURE
                        Temperature during sampling for generation.
  --top-k TOP_K         Top K during sampling for generation.
  --top-p TOP_P         Top P during sampling for generation.
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of tokens to generate.
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Order of tensor parallelism. Defaults to 1.
  --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE
                        Order of pipeline parallelism. Defaults to 1.
  --context-parallel-size CONTEXT_PARALLEL_SIZE
                        Order of context parallelism. Defaults to 1.
  --output-file OUTPUT_FILE
                        Output file containing the generated text produced by the Evo2 model. If not provided, the output will be logged.
```

As in `train_evo2`, `--ckpt-dir` points to the NeMo2 checkpoint directory for Evo2 that you want to load for inference. `--output-file` can be used to dump the output into a `.txt` file, and if not specified the output will be logged in the terminal.

```
[NeMo I 2025-01-06 17:22:22 infer:102] ['CTCTTCTGGTATTTGG']
```

## Prediction

To run a forward pass of the Evo2 model, you can call `predict_evo2`, which processes a batch of sequences and returns either raw token logits or, if `--output-log-prob-seqs` is set, log-probability scores.

For example, to predict the log-probability scores of a batch of sequences saved to `fasta_path`, you can run the following command:

```bash
predict_evo2 \
  --fasta <fasta_path> \
  --ckpt-dir <PATH_TO_CHECKPOINT> \
  --output-dir <PATH_TO_OUTPUT_FILE> \
  --model-size 1b \
  --tensor-parallel-size 1 \
  ----pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --output-log-prob-seqs
```

An example of using `predict_evo2` for variant effect prediction can be found in our [Evo 2 Zeroshot BRCA1 Notebook](https://docs.nvidia.com/bionemo-framework/latest/user-guide/examples/bionemo-evo2/evo2_zeroshot_brca). This notebook demonstrates how to use Evo2 to predict whether single nucleotide variants (SNVs) in the BRCA1 gene are likely to be harmful to protein function and potentially increase cancer risk, by comparing the model's log probability scores between the reference and variant sequences.

## Checkpoint conversion from hugging face to NeMo2

The following conversion script should work on any savanna formatted arc evo2 checkpoint. Make sure you match up the
model size with the checkpoint you are converting.
The pyproject.toml makes the conversion script available as a command line tool `evo2_convert_to_nemo2`, so you
can try replacing:

```bash
evo2_convert_to_nemo2 \
  ...
```

with the following if you want to run with `-m pdb` or something:

```bash
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  ...
```

### 1b-8k

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_1b_base \
  --model-size 1b --output-dir nemo2_evo2_1b_8k
```

This new checkpoint `nemo2_evo2_1b_8k` is ready to go in nemo2 format in downstream pretraining or prediction workflows.

#### Optional steps if you want to register the model with NGC

If you want to register the checkpoint with NGC (typically only NVIDIA employees) then you can do the following.

To create the checkpoint for distribution in NGC, first cd into the checkpiont directory:

```bash
cd nemo2_evo2_1b_8k
```

Then run the following command to make a tar of the full directory that gets unpacked into the current directory which
our NGC loader expects:

```bash
tar -czvf ../nemo2_evo2_1b_8k.tar.gz .
```

Finally `sha256sum` the tar file to get the checksum:

```bash
sha256sum nemo2_evo2_1b_8k.tar.gz
```

Then register it into the loader for testing purposes by editing
`sub-packages/bionemo-core/src/bionemo/core/data/resources/evo2.yaml`.

### 7b-8k

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_7b_base \
  --model-size 7b --output-dir nemo2_evo2_7b_8k
```

### 7b-1M

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_7b \
  --model-size 7b_arc_longcontext --output-dir nemo2_evo2_7b_1m
```

### 40b-8k

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_40b_base \
  --model-size 40b --output-dir nemo2_evo2_40b_8k
```

### 40b-1M

```bash
evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_40b \
  --model-size 40b_arc_longcontext --output-dir nemo2_evo2_40b_1m
```
