# CodonFM: Foundation Models for Codon Sequences

CodonFM is a fully open-source suite of foundation models trained directly on codon sequences to learn contextual codon representations and enable downstream codon-aware tasks. We release the entire stack: code, training/finetuning/evaluation scripts, dockerized environments, experiment templates, and pre-trained model weights under an open license for transparent and reproducible use. 

Our primary model families include:
- **Encodon**: Masked language modeling (bidirectional) over codons with scalable architectures (80M to 1B)
- **Decodon**: Autoregressive (GPT-style) language modeling over codons with organism conditioning (200M to 1B)

Public links to the pre-trained checkpoints: [80M](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1), [600M](https://huggingface.co/nvidia/NV-CodonFM-Encodon-600M-v1), [1B](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1), [1B-Cdwt](https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1). 

The checkpoints can also be found on NGC [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/nv_codonfm_encodon).

## Methodology and Results

The pre-print of this work with detailed methodology and results can be found [here](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf)

If you like this work please cite it as follows:
```bibtex
@article{codonfm_2025,
author = {Darabi+, Sajad and Cao+, Fan and Naghipourfar+, Mohsen and Rabi, Sara and Sethia, Ankit and Gion, Kyle and Grewal, Jasleen and Cohen, Jonathan and Greenleaf, William and Goodarzi*, Hani and Sundaram*, Laksshman},
title = {{Learning the language of codon translation with CodonFM}},
url = {https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf},
year = {2025}
}
```
Note: Sajad Darabi, Fan Cao and Mohsen Naghipourfar are equal contributing first authors.

Corresponding Author: Hani Goodarzi and Laksshman Sundaram

## Accelerated CodonFM
This repository contains the exact code used in the [pre-print](#methodology-and-results). 

An accelerated version of the codebase is available in [BioNeMo Framework Recipes](https://github.com/NVIDIA/bionemo-framework/tree/main/bionemo-recipes/recipes/codonfm_ptl_te), which uses [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) to accelerate training and inference. Accelerated checkpoints are available for all Encodon model variants: [80M](https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-80M-v1), [600M](https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-600M-v1), [1B](https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-1B-v1), [1B-Cdwt](https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-Cdwt-1B-v1).


## Table of Contents
- [Pre-trained Models](#pre-trained-models)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Data](#data-)
- [Running Training/Finetuning/Evaluation](#running-trainingfinetuningevaluation)
- [Sequence Generation](#sequence-generation)
- [Codon Optimization](#codon-optimization)
- [Cluster Execution Setup](#cluster-execution-setup)
- [Using Wandb with CodonFM](#using-wandb-with-codonfm)
- [Testing](#testing)
- [License](#license)
- [Contact](#contact)

## Pre-trained Models

The table below summarizes the set of open source pre-trained weights currently made available.

### Encodon Models (Bidirectional)

Training scripts are in `experiment_scripts/pretraining/encodon_filtered/`.

| Model | Variant | Hidden size | Layers | Heads | Intermediate | Script | Checkpoint |
|---|---|---|---|---|---|---|---|
| Encodon 80M | MLM (random p=0.15) | 1024 | 6 | 8 | 4096 | `mlm/encodon_80m.sh` | [link](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1) |
| Encodon 600M | MLM (random p=0.15) | 2048 | 12 | 16 | 8192 | `mlm/encodon_600m.sh` | [link](https://huggingface.co/nvidia/NV-CodonFM-Encodon-600M-v1) |
| Encodon 1B | MLM (random p=0.15) | 2048 | 18 | 16 | 8192 | `mlm/encodon_1b.sh` | [link](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1) |
| Encodon 1B (CDSWT) | MLM (codon frequency-weighted) | 2048 | 18 | 16 | 8192 | `cdswt/encodon_1b.sh` | [link](https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1) |

### Decodon Models (Autoregressive)

Training scripts are in `experiment_scripts/pretraining/decodon/filtered/`.

| Model | Variant | Hidden size | Layers | Heads | Intermediate | Script |
|---|---|---|---|---|---|---|
| Decodon 200M | CLM with organism conditioning | 1024 | 16 | 16 | 4096 | `decodon_200m.sh` |
| Decodon 1B | CLM with organism conditioning | 2048 | 18 | 16 | 8192 | `decodon_1b.sh` |

## Repository Structure
High-level overview (NerdTree-style):

```
codon-fm/
├── src/ — core library and CLI entrypoints
│   ├── runner.py — entry for pretrain/finetune/eval
│   ├── config.py — model/data/trainer configs
│   ├── tasks.py — pretraining/finetuning/eval tasks
│   ├── models/ — model definitions and components
│   │   ├── encodon_pl.py — Encodon Lightning module
│   │   └── decodon_pl.py — Decodon Lightning module
│   ├── data/ — datamodules, datasets, preprocessing
│   │   └── preprocess/ — item level process items
│   ├── inference/ — inference wrappers and prediction definitions
│   │   ├── encodon.py — Encodon inference with generation
│   │   ├── decodon.py — Decodon inference with generation
│   │   └── generation_utils.py — GA utilities for optimization
│   ├── tokenizer/ — codon tokenizer and mappings
│   └── utils/ — logging, schedulers, writers, helpers
├── scripts/ — codon optimization and utility scripts
│   ├── codon_optimize.py — main optimization pipeline
│   ├── optimize_encodon.sh — Encodon optimization wrapper
│   └── optimize_decodon.sh — Decodon optimization wrapper
├── experiment_scripts/ — launch scripts for pre-training
│   └── pretraining/ — Encodon and Decodon pretraining
├── data_scripts/ — data download and curation tools
├── notebooks/ — analysis and evaluation notebooks
├── env.example — sample env vars
└── README.md — repo guide
```

## Quickstart
To run the scripts in this repository, we recommend using the provided Docker setup.

### 1. Clone the repository
```bash
git clone https://github.com/NVIDIA-Digital-Bio/CodonFM
cd codon-fm
```
### 2. Docker Setup

The fastest way to get up and running with CodonFM is through the Docker setup below. This is an interactive development environment, you can build and launch a container that mounts your local repository. This allows you to edit code locally and run it inside the container.

To build and launch the development container, simply run the following from the root folder:

```bash
bash run_dev.sh
```

This script will:
1.  Build the development Docker image using the `development` target in the `Dockerfile`.
2.  Pass your user and group IDs to the container to avoid permission issues with mounted files.
3.  Stop and remove any existing container with the same name.
4.  Launch a new container with your local code mounted at `/workspace`, GPU access, host networking, and common directories for data and SSH keys.

You can also customize the data and checkpoint directory paths by passing arguments:
```bash
bash run_dev.sh --data-dir /path/to/your/data --checkpoints-dir /path/to/your/checkpoints
```

You will be dropped into a `bash` shell inside the container as a non-root user.

#### Evaluation Notebooks 📓

A series of notebooks are provided in the `notebooks` directory show casing multiple use cases such as zero-shot variant prediction and finetuning on downstream tasks. See a brief overview below:

| Notebook | Description |
|---|---|
| [00-Mutation-Datasets-Preprocessing.ipynb](notebooks/00-Mutation-Datasets-Preprocessing.ipynb) | Prepare and harmonize mutation datasets used across evaluations. |
| [0-Zero-Shot-Mutation-Variant-CancerHotspot.ipynb](notebooks/0-Zero-Shot-Mutation-Variant-CancerHotspot.ipynb) | Zero-shot variant effect scoring on Cancer Hotspots. |
| [1-Zero-Shot-Mutation-Variant-DDD-ASD.ipynb](notebooks/1-Zero-Shot-Mutation-Variant-DDD-ASD.ipynb) | Zero-shot scoring on Deciphering Developmental Disorders (DDD) and autism spectrum disorder (ASD) cohort study, which catalogs genetic mutations linked to rare pediatric and developmental diseases, to evaluate separation of healthy versus disease coh on coding sequence context.|
| [2-Zero-Shot-Mutation-Variant-Clinvar-Alphamissense.ipynb](notebooks/2-Zero-Shot-Mutation-Variant-Clinvar-Alphamissense.ipynb) | Zero-shot evaluation on ClinVar missense variants classifying benign vs. pathogenic |
| [3-Zero-Shot-Mutation-Variant-Clinvar-Synonymous.ipynb](notebooks/3-Zero-Shot-Mutation-Variant-Clinvar-Synonymous.ipynb) | Zero-shot evaluation on ClinVar synonymous variants evaluating how the models separate benign versus pathogenic synonymous mutations.|
| [4-EnCodon-Downstream-Task-riboNN.ipynb](notebooks/4-EnCodon-Downstream-Task-riboNN.ipynb) | Predicts ribosome profiling signal intensity along coding sequences, evaluating how well models capture translation efficiency and codon-level regulation from sequence context. |
| [5-EnCodon-Downstream-Task-mRFP-expression.ipynb](notebooks/5-EnCodon-Downstream-Task-mRFP-expression.ipynb) | Predicts fluorescent protein expression levels (mRFP) from coding sequences, testing how accurately models capture codon-dependent effects on translation efficiency and protein abundance.|
| [6-EnCodon-Downstream-Task-mRNA-stability.ipynb](notebooks/6-EnCodon-Downstream-Task-mRNA-stability.ipynb) | Predicts mRNA stability from coding sequences evaluating how the models associate codon composition with stability of mRNA.|


### Data 📊

#### Pre-training Dataset

The data curation tools live under `data_scripts/data_curation/`.

- Main entrypoint: open and run `data_scripts/data_curation/download_cds_clean.ipynb`. It documents how to obtain coding sequences (CDS), process metadata, and produce curated outputs.
- Filtering resources: `data_scripts/data_curation/taxids_to_remove_bac.json` lists bacterial taxids to exclude during curation.
- Recommended environment: use the provided dev container (`bash run_dev.sh`), then open the notebook in Jupyter/VS Code and execute the cells.

Outputs from the notebook (cleaned CDS files and metadata tables) can be transformed into training-ready formats memmap creation script in `src/data/data_scripts/ncbi_memmap_dataset_batched.py` on the output of the `src/data/data_curation/` notebook. This can then be consumed by`CodonMemmapDataset`.

#### Evaluation Datasets

- mRFP expression and mRNA stability:
  - Open and run the notebooks `notebooks/5-EnCodon-Downstream-Task-mRFP-expression.ipynb` and `notebooks/6-EnCodon-Downstream-Task-mRNA-stability.ipynb`. These notebooks contain cells that download/prepare the datasets and guide you through executing the evaluations end-to-end.
- Mean translation efficiency prediction task:
  - Open and run the notebook `notebooks/4-EnCodon-Downstream-Task-riboNN.ipynb`. It will download/prepare the downstream dataset and guide you through finetuning on this downstream task.
- Synonymous, DDD/ASD, and Cancer Hotspot variant datasets:
  - Follow `notebooks/00-Mutation-Datasets-Preprocessing.ipynb`. This notebook includes a cell that lists the required input files (with expected names/locations) and outlines how to process them into harmonized formats.
  - After preprocessing, use the task-specific notebooks in `notebooks/` (e.g., `0-...CancerHotspot.ipynb` and `1-...DDD-ASD.ipynb`) which consume the harmonized outputs produced by the preprocessing notebook.

### Running Training/Finetuning/Evaluation
The main entry point is `src/runner.py` which supports three modes:

#### Pre-training

The explicit scripts used to train the released checkpoints are referenced in [Pre-trained Models](#pre-trained-models)

```bash
python -m src.runner pretrain \
    --out_dir <output_dir> \
    --exp_name <experiment_name> \
    --model_name <model_size> \
    --data_path <path_to_data> \
    --process_item mlm_memmap \
    --dataset_name CodonMemmapDataset \
    --lr <learning_rate> \
    --num_gpus <num_gpus> \
    --num_nodes <num_nodes>
```

Optional path overrides:
```bash
  --out_dir <dir>
  --checkpoints_dir <dir>
  --pretrained_ckpt_path <path>
```

**Available `--model_name` options:**
- Encodon: `encodon_80m`, `encodon_600m`, `encodon_1b`, `encodon_5b`, `encodon_10b`
- Decodon: `decodon_200m`, `decodon_1b`

**Available `--process_item` options:**
- `mlm_memmap`: Constructs MLM training examples using memory-mapped data input format (Encodon).
- `clm_memmap`: Constructs CLM (causal language modeling) examples for autoregressive training (Decodon).
- `mutation_pred_mlm`: Constructs mutation prediction scoring input for the model using ref/alt/mut pos.
- `mutation_pred_clm`: Constructs mutation prediction for autoregressive models.
- `mutation_pred_likelihood`: Constructs input sentence with alt mutation at input to be scored by the model.
- `codon_sequence`: Constructs a codon sequence that can be inputed into the model.
- `generation_prompt`: Used for sequence generation tasks.

**Available `--dataset_name` options:**
- `CodonMemmapDataset`: dataset to support memory-mapped pre-training dataset used for pre-training
- `MutationDataset`: dataset for mutation prediction
- `CodonBertDataset`: dataset to ingest codon sequences.
- `GenerationPromptDataset`: dataset for sequence generation tasks.

#### Fine-tuning
The publicly available checkpoints can be finetuned using the finetuning options.

**Available finetuning options:**
- `lora`: Fine-tunes low-rank adapters within a pretrained model added to each transformer layer to reduce training cost and memory usage.
- `head_only_random`: Trains a randomly initialized output head while the remainder of the model is kept frozen.
- `head_only_pretrained`: Trains a pretrained output head while the remainder of the model is kept frozen.
- `full`: Fine-tunes all parameters of the model end-to-end

This is an example commandline for running finetuning:

```bash
python -m src.runner finetune \
    --out_dir <output_dir> \
    --exp_name <experiment_name> \
    --model_name <model_size> \
    --pretrained_ckpt_path <path_to_pretrained_checkpoint> \
    --data_path <path_to_data> \
    --process_item <process-item-to-use> \
    --dataset_name <dataset-name> \
    --finetune_strategy <strategy>
```

#### Evaluation
The publicly available checkpoints can be used to launch evaluation runs as well.

**Available tasks:**
- `mutation_prediction`: Scores a specified mutation via ref-vs-alt codon log-likelihood ratio.
- `masked_language_modeling`: Predicts masked codon tokens from surrounding sequence context.
- `fitness_prediction`: Estimates sequence fitness as the mean log-likelihood of the sequence as predicted by the model.
- `embedding_prediction`: Extracts encoder CLS embeddings for each input.
- `downstream_prediction`: Uses the downstream cross-attention head for task-specific classification/regression.
- `next_codon_prediction`: Predicts the next codon in autoregressive mode (Decodon).
- `sequence_generation`: Generates full codon sequences from prompts.

This is an example commandline for running evaluation:

```bash
python -m src.runner eval \
    --out_dir <output_dir> \
    --exp_name <experiment_name> \
    --model_name <model_size> \
    --checkpoint_path <path_to_checkpoint> \
    --data_path <path_to_data> \
    --task_type <task_type> \
    --predictions_output_dir <output_directory>
```

## Sequence Generation

CodonFM supports sequence generation for both Encodon (bidirectional) and Decodon (autoregressive) models.

### Encodon Generation Modes

**Bidirectional (Mask-and-Predict)**:
- Iteratively masks and predicts positions with temperature annealing
- Best for diverse, high-quality sequence generation
- Supports batch processing

**Autoregressive**:
- Left-to-right generation with synonymous codon constraints
- Uses beam search for optimal sequence selection

### Decodon Generation

- Native autoregressive (GPT-style) generation
- Organism conditioning for species-specific optimization
- Supports temperature sampling and beam search

## Codon Optimization

CodonFM includes a comprehensive codon optimization pipeline that combines model-guided sequence generation with a Genetic Algorithm (GA) for multi-objective optimization.

### Usage Examples

#### Encodon Optimization (Bidirectional)

```bash
./scripts/optimize_encodon.sh \
    --aa "MVSKGEELFTGVVPILVELDGDVNGHKFSVSG..." \
    --ckpt /path/to/encodon_checkpoint.ckpt \
    --mode bidir \
    --organism human \
    --beam-width 10 \
    --ga-generations 50 \
    --out-json ./optimized_output.json
```

#### Decodon Optimization (Autoregressive)

```bash
./scripts/optimize_decodon.sh \
    --aa "MVSKGEELFTGVVPILVELDGDVNGHKFSVSG..." \
    --ckpt /path/to/decodon_checkpoint.ckpt \
    --organism human \
    --beam-width 10 \
    --ga-generations 50 \
    --out-json ./optimized_output.json
```

### Configuration Options

**Generation Parameters:**
- `--beam-width`: Number of beam candidates (default: 10)
- `--mask-ratio`: Fraction of positions to mask per iteration (bidirectional, default: 0.2)
- `--parallel-iterations`: Number of mask-and-predict iterations (default: 20)
- `--temperature-start/end`: Temperature annealing range (default: 1.2 → 0.5)

**GA Parameters:**
- `--ga-generations`: Number of GA generations (default: 50, set to 0 to disable)
- `--ga-population-size`: Population size (default: 100)
- `--ga-crossover-rate`: Crossover probability (default: 0.7)
- `--ga-mutation-rate`: Mutation probability (default: 0.02)

**Fitness Weights:**
All weights are configurable via `--weight-<metric>` flags.

## Cluster Execution Setup

This project uses `nemo_run` to simplify the process of running experiments on a Slurm cluster. To configure your environment for cluster execution, please follow these steps:

1.  **Create an environment file**:
    Copy the example environment file to a new `.env` file in the root of the project:
    ```bash
    cp env.example .env
    ```

2.  **Configure your settings**:
    Open the `.env` file and fill in the required values. These variables are used by the `nemo_run` executors to connect to the cluster and manage your jobs.

    ```bash
    # ------------------NEMO RUN CONFIGURATION------------------
    # The following variables are used for configuring nemo_run executors for cluster execution.
    # This setup allows for user-specific configurations without modifying the source code.

    # User name for cluster login.
    NEMORUN_USER=your_username

    # Absolute path to the SSH key for cluster access.
    NEMORUN_SSH_KEY_PATH=/path/to/your/ssh/key

    # Default project name for organizing experiments.
    NEMORUN_PROJECT_NAME=codon-fm

    # Base directory on the cluster for storing results. This path should be accessible by the user on the cluster.
    # Example: /lustre/fsw/portfolios/healthcareeng/users/<your_user>/results
    NEMORUN_RESULTS_DIR=/path/to/your/results_dir

    # Directory on the cluster where the data is located.
    # Example: /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_virtualcell/data/codonfm
    NEMORUN_DATA_DIR=/path/to/your/data_dir

    # Default Docker container image to be used for the jobs.
    NEMORUN_CONTAINER_IMAGE="gitlab-master.nvidia.com/bio-foundation-models/codon-fm:latest"

    # Account name for cluster job submission. Defaults to 'healthcareeng_virtualcell' if not set.
    NEMORUN_ACCOUNT=healthcareeng_virtualcell
    ```

3.  **Source the environment file**:
    For the environment variables to be available in your shell session, you should source the `.env` file:
    ```bash
    source .env
    ```
    It is recommended to add this line to your `~/.bashrc` or `~/.zshrc` file to automatically load the variables when you start a new terminal session.

Once these steps are completed, you can use the cluster executors defined in `src/executors/cluster.py` to run your experiments.

## Using Wandb with CodonFM

To use Wandb with CodonFM, set your Weights & Biases API key for logging in the .env file:

```bash
# WANDB key (optional; only needed if enabling --enable_wandb)
WANDB_API_KEY=your_wandb_api_key
```
You can then source the .env file.

```bash
source .env
```

When launching runs, enable WandB logging by passing `--enable_wandb` and providing `--project_name` and `--entity`. If these are omitted, WandB logging will be skipped.


## Testing

This repository includes a comprehensive test suite to ensure code quality and functionality. It is **strongly recommended** to run the full test suite whenever:

- **Before pushing commits** to any branch
- **After merging** changes back into the main branch  
- **During development** to catch issues early

### Running Tests

To run the complete test suite:

```bash
# Run all tests
python -m pytest tests/

# Run tests with verbose output
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/data/          # Data-related tests
python -m pytest tests/models/        # Model tests
python -m pytest tests/tokenizer/     # Tokenizer tests
python -m pytest tests/utils/         # Utility tests
python -m pytest tests/inference/     # Inference tests
python -m pytest tests/scripts/       # Script tests
```

## License

Copyright @ 2025, NVIDIA Corporation. All rights reserved.
The source code is made available under Apache-2.0.
The model weights are made available under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Contact
Any questions or correspondance should be sent to nv-codonfm@nvidia.com
