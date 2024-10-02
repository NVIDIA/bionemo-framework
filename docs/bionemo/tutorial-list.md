# BioNeMo Framework Tutorials

The best way to get started with BioNeMo Framework is with the tutorials. Below are some of the example walkthroughs which contain code snippets that you can run from within the container.

Tutorials are presented as notebooks (`.ipynb` format), which may contain various code snippets in formats like Python, Bash, YAML, etc. You can follow the instructions in these files, make appropriate code changes, and execute them in the container.

It is convenient to first launch the BioNeMo Framework container and copy the tutorial files to the container, either via the JupyterLab interface drag-and-drop or by mounting the files during the launch of the container (`docker run -v ...`).

| Topic              | Title                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Model Pre-Training | [Launching a MegaMolBART model pre-training with ZINC-15 dataset](./notebooks/model_training_mmb.ipynb) |
| Custom Datasets | [Setting up the ZINC15 dataset used for training MolMIM](./notebooks/ZINC15-data-preprocessing.ipynb) |
| Model Pre-Training | [Launching a MolMIM model pre-training with ZINC-15 dataset, both from scratch and starting from an existing checkpoint](./notebooks/model_training_molmim.ipynb) |
| Model Pre-Training | [ESM-1nv: Data preprocessing and model pre-training using BioNeMo with curated data from UniRef50, UniRef90](./notebooks/model_training_esm1nv.ipynb) |
| Model Pre-Training | [ESM-2nv: Data Preprocessing and Model Training](./notebooks/model_training_esm2nv.ipynb) |
| Model Pre-Training | [Pretraining a Geneformer model for representing single cell RNA-seq data](./notebooks/geneformer_cellxgene_tutorial.ipynb) |
| Geneformer Benchmarking| [Benchmarking pre-trained Geneformer models against a baseline with cell type classification](./notebooks/Geneformer-celltype-classification-example.ipynb) |
| Model Training     | [Launching an EquiDock model pre-training with DIPS or DB5 datasets](./notebooks/model_training_equidock.ipynb)|
| Inference          | [Performing Inference with MegaMolBART for Generative Chemistry and Predictive Modeling with RAPIDS](./notebooks/MMB_GenerativeAI_Inference_with_examples.ipynb) |
| Inference          | [Zero-Shot Protein Design Using ESM-2nv](./notebooks/esm2nv-mutant-design.ipynb) |
| Inference          | [Performing Inference with ESM-2nv and Predictive Modeling with RAPIDS](./notebooks/protein-esm2nv-clustering.ipynb) |
| Inference          | [MolMIM Inferencing for Generative Chemistry and Downstream Prediction](./notebooks/MolMIM_GenerativeAI_local_inference_with_examples.ipynb) |
| Inference          | [Performing Property-guided Molecular Optimization with MolMIM, which internally involves inference](./notebooks/cma_es_guided_molecular_optimization_molmim.ipynb) |
| Inference          | [Performing inference and cell clustering on CELLxGENE data with a pretrained geneformer model](./notebooks/geneformer_cellxgene_pretrained_inference_tutorial.ipynb) |
| Inference          | [Performing inference on OAS sequences with ESM-2nv](./notebooks/esm2_oas_inferencing.ipynb) |
| Model Finetuning   | [Overview of Finetuning pre-trained models in BioNeMo](./notebooks/bionemo-finetuning-overview.ipynb)                             |
| Model Finetuning   | [Fine-Tune ESM-2nv on FLIP Data for Sequence-Level Classification, Regression, Token-Level Classification, and with LoRA Adapters](./notebooks/esm2_FLIP_finetuning.ipynb) |
| Model Pre-Training and Finetuning   | [Pretrain from Scratch, Continue Training from an Existing Checkpoint, and Fine-tune ESM-2nv on Custom Data](./notebooks/esm2_paratope_finetuning.ipynb) |
| Encoder Finetuning | [Encoder Fine-tuning in BioNeMo: MegaMolBART](./notebooks/encoder-finetuning-notebook-fw.ipynb)                             |
| Downstream Tasks   | [Training a Retrosynthesis Model using USPTO50 Dataset](./notebooks/retrosynthesis-notebook.ipynb)                             |
| Downstream Tasks   | [Fine-tuning MegaMolBART for Solubility Prediction](./notebooks/physchem-notebook-fw.ipynb)                                 |
| Custom Datasets    | [Adding the OAS Dataset: Downloading and Preprocessing](./notebooks/custom-dataset-preprocessing-fw.ipynb) |
| Custom Datasets    | [Adding the OAS Dataset: Modifying the Dataset Class](./notebooks/custom-dataset-class-fw.ipynb) |
| Custom DataLoaders | [Creating a Custom Dataloader](./notebooks/custom-dataset-dataloader.ipynb) |
| Inference          | [Creating and Visualizing Embeddings with DNABERT](./notebooks/dnabert_inference) |
| Model Finetuning   | [Pretrain, Fine-tune, and Perform Inference with DNABERT for Splice Site Prediction](./notebooks/dnabert_pretrain_finetune) |
