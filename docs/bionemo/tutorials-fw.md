# BioNeMo Framework Tutorials

The best way to get started with BioNeMo Framework is with the tutorials. Below are some of the example walk-throughs/tutorials which contains code snippets that you can run from within the BioNeMo container. 

Some of the tutorials are presented in the format of Markdown with Python or shell codes, where a user can run example cells by copying the code to a script in an appropriate folder inside the BioNeMo container and execute it.

Other tutorial files are provided in the MarkDown (``.md``) format, where the file may contain various example code snippets in YAML/bash/python/etc. formats. You can follow the instructions provided in those files, make appropriate changes to the codes, and execute them once the container is launched. 

In both types of tutorials, it is convenient to first launch the BioNeMo Framework container, and copy the tutorial files to the container -- either via the Jupyter-Lab interface drag-and-drop, or by mounting the files during the launch of container (```docker run -v ...```)



| Topic              | Title                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Model Pre-Training | [Launching a MegaMolBART model pre-training with ZINC-15 dataset](./notebooks/model_training_mmb.ipynb) |
| Model Pre-Training | [Launching an ESM-1nv model pre-training with UNiRef50 dataset](./notebooks/model_training_esm1nv.ipynb) |
| Inference          | [Performing Inference with MegaMolBART for Generative Chemistry and Predictive Modeling with RAPIDS](./notebooks/MMB_GenerativeAI_Inference_with_examples.ipynb) |
| Inference          | [Performing Inference with ESM1-nv and Predictive Modeling with RAPIDS](./notebooks/protein-esm1nv-clustering.ipynb) |
| Model Finetuning   | [Overview of Finetuning pre-trained models in BioNeMo](./notebooks/bionemo-finetuning-overview.ipynb)                             |
| Encoder Finetuning | [Encoder Fine-tuning in BioNeMo: MegaMolBART](./notebooks/encoder-finetuning-notebook-fw.ipynb)                             |
| Downstream Tasks   | [Training a Retrosynthesis Model using USPTO50 Dataset](./notebooks/retrosynthesis-notebook.ipynb)                             |
| Downstream Tasks   | [Fine-tuning MegaMolBART for Solubility Prediction](./notebooks/physchem-notebook-fw.ipynb)                                 |
| Custom Datasets    | [Adding the OAS Dataset: Downloading and Preprocessing](./notebooks/custom-dataset-preprocessing-fw.ipynb) |
| Custom Datasets    | [Adding the OAS Dataset: Modifying the Dataset Class](./notebooks/custom-dataset-class-fw.ipynb) |
| Custom DataLoaders | [Creating a Custom Dataloader](./notebooks/custom-dataset-dataloader.ipynb) |
