# Next Steps

Now that you have successfully launched the Docker container and entered it, this section will guide you through the container, initial steps to take within the container (such as configuration, downloading pre-trained model weights, etc.), and where to find tutorials.

## NGC CLI Configuration

NVIDIA NGC Command Line Interface (CLI) is a command-line tool for managing Docker containers in NGC. If NGC is not already installed in the container, download it as per the instructions [here](https://org.ngc.nvidia.com/setup/installers/cli) (note that within the container, the AMD64 Linux version should be installed).

Once installed, run `ngc config set` to establish NGC credentials within the container.

## First-Time Setup

First, invoke the following launch script. The first time, it will create a .env file and exit:

```bash
./launch.sh
```

Next, edit the .env file with the correct NGC parameters for your organization and team:

```bash
    NGC_CLI_API_KEY=<YOUR_API_KEY>
    NGC_CLI_ORG=<YOUR_ORG>
    NGC_CLI_TEAM=<YOUR_TEAM>
```

## Download Model Weights

You may now download all pre-trained model checkpoints from NGC through the following command:

```bash
./launch.sh download
```
This will download all models to the `workspace/bionemo/models` directory. Optionally, you may persist the models by copying them to your mounted workspace, so that they need not be redownloaded each time.

## Directory Structure

Note that `workspace/bionemo` is the home directory for the container. Below are a few key components:
* `bionemo`: Contains the core BioNeMo package, which includes base classes for BioNeMo data modules, tokenizers, models, etc.
* `examples`: Contains example scripts, datasets, YAML files, and notebooks
* `models`: Contains all pre-trained models checkpoints in .nemo format.

## Weights and Biases Setup (Optional)

Training progress and charts of the models can be visualized through [Weights and Biases](https://docs.wandb.ai/guides/track/public-api-guide). Setup your [API Key](https://docs.wandb.ai/guides/track/public-api-guide#authentication) to enable logging.

## BioNeMo Framework Tutorials

The best way to get started with BioNeMo Framework is with the tutorials. Below are some of the example walkthroughs which contain code snippets that you can run from within the container.

Tutorials are presented as notebooks (`.ipynb` format), which may contain various code snippets in formats like Python, Bash, YAML, etc. You can follow the instructions in these files, make appropriate code changes, and execute them in the container.

It is convenient to first launch the BioNeMo Framework container and copy the tutorial files to the container, either via the JupyterLab interface drag-and-drop or by mounting the files during the launch of the container (`docker run -v ...`).

| Topic              | Title                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Model Pre-Training | [Launching a MegaMolBART model pre-training with ZINC-15 dataset](./notebooks/model_training_mmb.ipynb) |
| Custom Datasets | [Setting up the ZINC15 dataset used for training MolMIM](./notebooks/ZINC15-data-preprocessing.ipynb) |
| Model Pre-Training | [Launching a MolMIM model pre-training with ZINC-15 dataset, both from scratch and starting from an existing checkpoint](./notebooks/model_training_molmim.ipynb) |
| Model Pre-Training | [Launching an ESM-1nv model pre-training with UniRef50 dataset](./notebooks/model_training_esm1nv.ipynb) |
| Model Pre-Training | [Launching an ESM-2nv model pre-training with curated data from UniRef50, UniRef90](./notebooks/model_training_esm2nv.ipynb) |
| Model Pre-Training | [Pretraining a geneformer model for representing single cell RNA-seq data](./notebooks/geneformer_cellxgene_tutorial.ipynb) |
| Geneformer Benchmarking| [Benchmarking pre-trained Geneformer models against a baseline with cell type classification](./notebooks/Geneformer-celltype-classification-example.ipynb) |
| Model Training     | [Launching an EquiDock model pre-training with DIPS or DB5 datasets](./notebooks/model_training_equidock.ipynb)|
| Inference          | [Performing Inference with MegaMolBART for Generative Chemistry and Predictive Modeling with RAPIDS](./notebooks/MMB_GenerativeAI_Inference_with_examples.ipynb) |
| Inference          | [Performing Inference with MolMIM for Generative Chemistry and Predictive Modeling with RAPIDS](./notebooks/MolMIM_GenerativeAI_local_inference_with_examples.ipynb) |
| Inference          | [Performing Inference with ESM1-nv and Predictive Modeling with RAPIDS](./notebooks/protein-esm1nv-clustering.ipynb) |
| Inference          | [Performing Inference with ESM2-nv and Predictive Modeling with RAPIDS](./notebooks/protein-esm2nv-clustering.ipynb) |
| Inference          | [Performing Property-guided Molecular Optimization with MolMIM, which internally involves inference](./notebooks/cma_es_guided_molecular_optimization_molmim.ipynb) |
| Inference          | [Performing inference and cell clustering on CELLxGENE data with a pretrained geneformer model](./notebooks/geneformer_cellxgene_pretrained_inference_tutorial.ipynb) |
| Model Finetuning   | [Overview of Finetuning pre-trained models in BioNeMo](./notebooks/bionemo-finetuning-overview.ipynb)                             |
| Encoder Finetuning | [Encoder Fine-tuning in BioNeMo: MegaMolBART](./notebooks/encoder-finetuning-notebook-fw.ipynb)                             |
| Downstream Tasks   | [Training a Retrosynthesis Model using USPTO50 Dataset](./notebooks/retrosynthesis-notebook.ipynb)                             |
| Downstream Tasks   | [Fine-tuning MegaMolBART for Solubility Prediction](./notebooks/physchem-notebook-fw.ipynb)                                 |
| Custom Datasets    | [Adding the OAS Dataset: Downloading and Preprocessing](./notebooks/custom-dataset-preprocessing-fw.ipynb) |
| Custom Datasets    | [Adding the OAS Dataset: Modifying the Dataset Class](./notebooks/custom-dataset-class-fw.ipynb) |
| Custom DataLoaders | [Creating a Custom Dataloader](./notebooks/custom-dataset-dataloader.ipynb) |
| Model Pre-Training| [Training DNABERT on a Custom Dataset](./notebooks/dnabert_pretrain_custom.ipynb) | 
