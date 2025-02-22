# bionemo-evo2

`bionemo-evo2` is a `pip`-installable package that contains **data preprocessing**, **training**, and **inferencing** code for Evo2, a new `Hyena`-based foundation model for genome generation and understanding. Built upon `Megatron-LM` parallelism and `NeMo2` algorithms, `bionemo-evo2` provides the remaining tools necessary to effectively fine-tune the pre-trained Evo2 model checkpoint on user-provided sequences at scale, and generate state-of-the-art life-like DNA sequences from Evo2 for downstream metagenomic tasks.

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
usage: train_evo2 [-h] -d DATASET_CONFIG [--num-nodes NUM_NODES] [--devices DEVICES] [--seq-length SEQ_LENGTH] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE] [--context-parallel-size CONTEXT_PARALLEL_SIZE] [--wandb-project WANDB_PROJECT] [--wandb-run-id WANDB_RUN_ID]
                  [--sequence-parallel] [--fp8] [--micro-batch-size MICRO_BATCH_SIZE] [--global-batch-size GLOBAL_BATCH_SIZE] [--grad-acc-batches GRAD_ACC_BATCHES] [--max-steps MAX_STEPS] [--val-check-interval VAL_CHECK_INTERVAL] [--grad-reduce-in-fp32] [--no-aligned-megatron-ddp] [--use-megatron-comm-overlap-llama3-8k] [--align-param-gather] [--straggler-detection] [--model-size {7b,40b,test}] [--experiment-dir EXPERIMENT_DIR] [--limit-val-batches LIMIT_VAL_BATCHES] [--ckpt-dir CKPT_DIR] [--restore-optimizer-from-ckpt] [--seed SEED] [--workers WORKERS] [--gc-interval GC_INTERVAL] [--enable-preemption] [--ckpt-async-save] [--nsys-profiling] [--nsys-start-step NSYS_START_STEP] [--nsys-end-step NSYS_END_STEP] [--nsys-ranks NSYS_RANKS [NSYS_RANKS ...]]

Train a Hyena model using NeMo 2.0.

options:
  -h, --help            show this help message and exit
  -d DATASET_CONFIG, --dataset-config DATASET_CONFIG
                        Path to the blended / weighted training dataset configuration YAML.
  --num-nodes NUM_NODES
                        Number of nodes to use for training, defaults to 1.
  --devices DEVICES     Number of devices to use for training, defaults to 1.
  --seq-length SEQ_LENGTH
                        Training sequence length
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Order of tensor parallelism. Defaults to 1.
  --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE
                        Order of pipeline parallelism. Defaults to 1.
  --context-parallel-size CONTEXT_PARALLEL_SIZE
                        Order of context parallelism. Defaults to 1.
  --wandb-project WANDB_PROJECT
                        Wandb project name
  --wandb-run-id WANDB_RUN_ID
                        Wandb run identifier
  --sequence-parallel   Set to enable sequence parallelism.
  --fp8                 Set to enable FP8
  --micro-batch-size MICRO_BATCH_SIZE
                        Micro-batch size for data-parallel training.
  --global-batch-size GLOBAL_BATCH_SIZE
                        Global batch size for training. If set to None, infer it from the TP, CP, and PP parameters.
  --grad-acc-batches GRAD_ACC_BATCHES
                        Number of batches to accumulate gradients over.
  --max-steps MAX_STEPS
                        Number of training optimizer update steps.
  --val-check-interval VAL_CHECK_INTERVAL
                        Number of steps between validation measurements and model checkpoints.
  --grad-reduce-in-fp32
                        Gradient reduce in FP32.
  --no-aligned-megatron-ddp
                        Do not do aligned gradient updates etc.
  --use-megatron-comm-overlap-llama3-8k
  --align-param-gather
  --straggler-detection
  --model-size {7b,40b,test}
                        Model size, choose between 7b, 40b, or test (4 layers, less than 1b).
  --experiment-dir EXPERIMENT_DIR
                        Directory to write model checkpoints and results to.
  --limit-val-batches LIMIT_VAL_BATCHES
                        Number of validation steps
  --ckpt-dir CKPT_DIR   Directory to restore an initial checkpoint from. Use this for supervised fine-tuning.
  --restore-optimizer-from-ckpt
                        Restore optimizer state from initial checkpoint. Defaults to False.
  --seed SEED           Set random seed for training.
  --workers WORKERS     Number of workers to use for data loading.
  --gc-interval GC_INTERVAL
                        Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps.
  --enable-preemption   Enable preemption hooks. If enabled this will save a checkpoint whenver slurm exits.
  --ckpt-async-save
  --nsys-profiling      Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop [regular python
                        command here]`
  --nsys-start-step NSYS_START_STEP
                        Start nsys profiling after this step.
  --nsys-end-step NSYS_END_STEP
                        End nsys profiling after this step.
  --nsys-ranks NSYS_RANKS [NSYS_RANKS ...]
                        Enable nsys profiling for these ranks.
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

## Checkpoint conversion from hugging face to NeMo2
The following conversion script should work on any savanna formatted arc evo2 checkpoint. Make sure you match up the
model size with the checkpoint you are converting.
The pyproject.toml also makes the conversion script available as a command line tool `evo2_convert_to_nemo2`, so you
can try replacing:
```bash
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  ...
```
with:
```bash
evo2_convert_to_nemo2 \
  ...
```


```bash
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  --model-path hf://arcinstitute/savanna_evo2_1b_base \
  --model-size 1b --output-dir nemo2_evo2_1b_8k
```

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
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  --model-path hf://arcinstitute/savanna_evo2_7b_base \
  --model-size 7b --output-dir nemo2_evo2_7b_8k
```
### 7b-1M
```bash
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  --model-path hf://arcinstitute/savanna_evo2_7b \
  --model-size 7b_arc_longcontext --output-dir nemo2_evo2_7b_1m
```
### 40b-8k
```bash
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  --model-path hf://arcinstitute/savanna_evo2_40b_base \
  --model-size 40b --output-dir nemo2_evo2_40b_8k
```
### 40b-1M
```bash
python \
  sub-packages/bionemo-evo2/src/bionemo/evo2/utils/checkpoint/convert_to_nemo.py \
  --model-path hf://arcinstitute/savanna_evo2_40b \
  --model-size 40b_arc_longcontext --output-dir nemo2_evo2_40b_1m
```
