# Release Notes
## BioNeMo Framework v1.9
### New Features
* [Documentation] Updated, executable ESM-2nv notebooks demonstrating: Data preprocessing and model training with custom datasets, Fine-tuning on FLIP data, Inference on OAS sequences, Pre-training from scratch and continuing training
* [Documentation] New notebook demonstrating Zero-Shot Protein Design Using ESM-2nv. Thank you to [@awlange](https://github.com/awlange) from A-Alpha Bio for contributing the original version of this recipe!

### Bug fixes and Improvements
* [Geneformer] Fixed bug in preprocessing due to a relocation of dependent artifacts.
* [Geneformer] Fixes bug in finetuning to use the newer preprocessing constructor.

## BioNeMo Framework v1.8
### New Features
* [Documentation] Updated, executable MolMIM notebooks demonstrating: Training on custom data, Inference and downstream prediction, ZINC15 dataset preprocesing, and CMA-ES optimization
* [Dependencies] Upgraded the framework to [NeMo v1.23](https://github.com/NVIDIA/NeMo/tree/v1.23.0), which updates PyTorch to version 2.2.0a0+81ea7a4 and CUDA to version 12.3.

### Bug fixes and Improvements
* [ESM2] Fixed a bug in gradient accumulation in encoder fine-tuning
* [MegaMolBART] Make MegaMolBART encoder finetuning respect random seed set by user
* [MegaMolBART] Finetuning with val_check_interval=1 bug fix

### Known Issues
* Minor training speed regression observed for models DNABERT, Geneformer, MolMIM
* Two known critical CVEs GHSA-cgwc-qvrx-rf7f, GHSA-mr7h-w2qc-ffc2. The vulnerabilities arise within a package that's installed by lightning by default. We do not use that package in bionemo framework container. we are also unable to remove the package in question as it's installed as a side-effect of installing lightning.
* Two known High CVEs from pytorch : GHSA-pg7h-5qx3-wjr3, GHSA-5pcm-hx3q-hm94.

## BioNeMo Framework v1.7
### New Models
* [DSMBind](https://www.biorxiv.org/content/10.1101/2023.12.10.570461v1), developed under the BioNeMo framework, is a model which can produce comparative values for ranking protein-ligand binding affinities. This release features the capability to perform inference using a newly trained checkpoint.
### New Features
* [EquiDock] Remove steric clashes as a post-processing step after equidock inference.
* [Documentation] Updated Getting Started section which sequentially describes prerequisites, BioNeMo Framework access, startup instructions, and next steps.

### Known Issues
* There is a known security vulnerability with NLTK that can allow for arbitrary code execution via pickle files that are external assets downloaded via nltk.download() (https://github.com/nltk/nltk/issues/3266). BioNeMo itself does not use this dependency in any way, however parts of NeMo text-to-speech (nemo.collections.tts) does use this vulnerable codepath. Since NeMo is installed in the BioNeMo release containers, users are urged to exercise caution when using  nemo.collections.tts or nltk.

## BioNeMo Framework v1.6
### New Features
* [Model Fine-tuning] `model.freeze_layers` fine-tuning config parameter added to freeze a specified number of layers. Thank you to github user [@nehap25](https://github.com/nehap25)!
* [ESM2]  Loading pre-trained ESM2 weights and continue pre-training on the MLM objective on a custom FASTA dataset is now supported.
* [OpenFold] MLPerf feature 3.2 bug (mha_fused_gemm) fix has merged.
* [OpenFold] MLPerf feature 3.10 integrated into bionemo framework.
* [DiffDock] Updated data loading module for DiffDock model training, changing from sqlite3 backend to webdataset.

## BioNeMo Framework v1.5
### New Models
* [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) is out of **Beta** status. This release includes newly trained checkpoints and benchmarks, including a variant based on the publication with 10M parameters, and the largest variant of geneformer publically available to date with 106M parameters.

## BioNeMo Framework v1.4
### New Models
* **Beta** [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) a foundation model for single-cell data that encodes each cell as represented by an ordered list of differentially expressed genes for that cell.

### New Features
* **Beta** [Geneformer pretraining with custom datasets](notebooks/geneformer_cellxgene_tutorial.ipynb)
* [Low-Rank Adaptation (LoRA) finetuning for ESM2](lora-finetuning-esm2.md)

### Bug fixes and Improvements
* [OpenFold training improved benchmarks and validation of optimizations](models/openfold.md)

### Known Issues
* BioNeMo Framework v24.04 container is vulnerable to [GHSA-whh8-fjgc-qp73](https://github.com/advisories/GHSA-whh8-fjgc-qp73) in onnx 1.14.0. Users are advised not to open untrusted onnx files with this image. Restrict your mount point to minimize directory traversal impact. A fix for this is scheduled in the 24.05 (May) release.

## BioNeMo Framework v1.3
### New Models
* MolMIM implementation under BioNeMo framework, [a small molecule model developed at NVIDIA](https://arxiv.org/abs/2208.09016) which can be used to produce embeddings and novel molecules.

### New Features
* [MolMIM](https://developer.nvidia.com/blog/new-models-molmim-and-diffdock-power-molecule-generation-and-molecular-docking-in-bionemo/) re-trained on more data is now available in the framework, and achieves [state of the art performance](models/molmim.md).
* [MolMIM property guided tutorial notebook](notebooks/cma_es_guided_molecular_optimization_molmim.ipynb) covering property guided optimization using our new framework model.
* [MolMIM training tutorial](notebooks/model_training_molmim.ipynb) available walking users through either training from scratch or from an existing checkpoint on your own data.
* [MolMIM tutorial notebook covering molecular sampling and property prediction](notebooks/MolMIM_GenerativeAI_local_inference_with_examples.ipynb) is also now available.
* Numerous optimizations from [NVIDIA's entry to the MLPerf competition](https://developer.nvidia.com/blog/optimizing-openfold-training-for-drug-discovery/) have been added to OpenFold. Documentation and detailed benchmarks are works in progress and will be published in upcoming releases. This release contains the following performance optimizations:
    * Fused GEMMs in multi-head attention (MHA)
    * Non-blocking data pipeline
    * BF16 precision training
    * Fused MHA gating
    * Inductor Compiled LayerNorm
    * OpenAI Triton LayerNorm kernels
    * OpenAI Triton MHA

### Bug fixes and Improvements
* NeMo upgraded to v1.22 ([see NeMo release notes](https://github.com/NVIDIA/NeMo/releases)),
* PyTorch Lightning upgraded to 2.0.7
* [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) has been removed from the release container. If users download models from inside the container (via e.g. `download_artifacts.py` or `launch.sh download`), the NGC CLI will be auto-installed to pull the models from NGC.

### Known Issues
* BioNeMo Framework v24.03 container is vulnerable to [GHSA-whh8-fjgc-qp73](https://github.com/advisories/GHSA-whh8-fjgc-qp73) in onnx 1.14.0. Users are advised not to open untrusted onnx files with this image. Restrict your mount point to minimize directory traversal impact.

## BioNeMo Framework v1.2
## New Models
* OpenFold implementation under BioNeMo framework, derived from public OpenFold and DeepMind AlphaFold-2.
* DNABERT implementation for computing embeddings for each nucleotide in the input DNA sequence.

### New Features
* Training recipes for DNABERT and OpenFold, including automated data processing and full configuration for training.
* Example tutorials for running inference using OpenFold.
* Splice Prediction downstream task example for DNABERT.
* Wrapper scripts for DNABERT and OpenFold to launch jobs on BCP.

### Bug fixes and Improvements
* Interface improvements for ESM2 data ingestion and pre-processing. The interface allows for explicit specification of training, validation, and test sets. The user may set `config.model.data.default_dataset_path` to maintain prior behavior, or set `config.model.data.train.dataset_path`, `config.model.data.val.dataset_path`, `config.model.data.test.dataset_path` which may all be unique.

### Known Issues
* OpenFold training speed does not yet include [MLPerf optimizations](https://blogs.nvidia.com/blog/scaling-ai-training-mlperf/), and these will be released in the subsequent release.

## BioNeMo Framework v1.1
## New Models
* EquiDock for protein-protein docking pose prediction
* DiffDock for protein-ligand blind docking pose generation

### New Features
* Training recipes for EquiDock and DiffDock, including automated data processing and full configuration for training.
* Accelerated inference and training for DiffDock via fast tensor-product kernels.
* Example tutorials for running inference using EquiDock and DiffDock.
* Recipes for running EquiDock and DiffDock on BCP and Slurm.
* Pipeline parallel supported for ESM-2nv.
* Migration of inference notebooks to using pytriton.

### Bug fixes and Improvements
* Faster pre-processing of data on BCP.
* Refactor of download_models.sh to download_models.py for easier CLI use.
* Refactor of install structure to move from /opt/nvidia to /workspace/bionemo. The environment variable $BIONEMO_HOME now points to the repo base and is required to be set for tests to pass.

### Security Notice

SchedMD Slurm in the release container is shipped with a security vulnerability, [CVE-2022-29501](https://ubuntu.com/security/CVE-2022-29501), and therefore this version of Slurm should not be used to run a Slurm cluster (specifically, the processes `slurmdbd`, `slurmctld`, and `slurmd`.

In general, the BioNeMo Framework release is designed to ship code and an environment that would be executed on local workstations, or deployed on clusters for large scale training jobs. This container is not designed to run as a service with public facing APIs. A full summary of security vulnerabilities can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework/security).

## BioNeMo Framework v1.0
## New Models
* ESM-2nv for protein sequence representations, pretrained weights of ESM-2 650M and ESM-2 3B converted from HF checkpoint available.

### New Features
* Pre-training recipes for ESM-2nv, including automated data processing and full configuration for training
* Fine-tuning of ESM-2nv with encoder frozen or trainable
* Downstream task finetuning support for single-value classification (e.g. subcellular localization), single-value regression (e.g. meltome) and per-token classification (e.g. secondary structure)
* Validation in loop to evaluate performance on downstream tasks during training
* Example tutorials for pre-training, fine tuning, and downstream tasks

## BioNeMo Framework v0.4.0
### New Models
* ESM-1nv for protein sequence representations, pretrained weights available
* ProtT5nv for protein sequence representation and sequence-to-sequence tasks, pretrained weights available
### New Features
* Pre-training for all models, including automated data processing and full configuration for training
* Fine-tuning of MegaMolBART, ESM-1nv, and ProtT5nv with encoder frozen or trainable
* Downstream task example applications – secondary structure prediction for ESM-1nv and ProtT5nv, physchem prediction (lipophilicity, FreeSolv, ESOL) and retrosynthesis prediction for MegaMolBART
* Validation in loop to evaluate performance on downstream tasks during training: physchem prediction (MegaMolBART) and secondary structure prediction (ESM-1nv and ProtT5nv).
* Pipeline parallelism supported as a beta feature. Not fully tested.
* Example notebooks for pre-training, fine tuning, and downstream tasks

### Known Issues
* Data preprocessing on DGX Cloud is slow. Faster to do it on a local machine.
### New APIs
* BioNeMoDataModule - Encapsulates dataset instantiation in bionemo models so that many different datasets can be used with the same model
* EncoderFineTuning - Base class to facilitate implementation of downstream tasks built on embeddings from other models
