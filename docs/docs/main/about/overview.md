# Overview of BioNeMo

BioNeMo is a software ecosystem produced by NVIDIA for the development and deployment of life sciences-oriented artificial intelligence models. The main components of BioNeMo are:

- **BioNeMo Recipes**: Self-contained, reproducible training recipes for biomolecular and language models. Each recipe bundles a HuggingFace-compatible model definition, training scripts, configuration, and sample data into a single directory that can be run independently. Recipes cover protein representation learning (ESM-2, AMPLIFY, Geneformer), DNA sequence modeling (Evo2), molecular generation, and general-purpose language models (Llama 3, Mixtral, Qwen). See the [recipes section](../recipes/index.md) for details.

- **BioNeMo Sub-package Utilities**: Lightweight, pip-installable Python packages that provide reusable building blocks for training and data processing. Key utilities include:

  - **bionemo-core** -- shared interfaces, data-loading helpers, and checkpoint management
  - **bionemo-moco** -- modular components for building diffusion and flow-matching generative models
  - **bionemo-noodles** -- fast FASTA/FASTQ parsing via a Python wrapper around the Rust [noodles](https://github.com/zaeleus/noodles) library
  - **bionemo-scdl** -- dataset classes optimized for single-cell data
  - **bionemo-size-aware-batching** -- memory-aware mini-batch construction for variable-length inputs
  - **bionemo-webdatamodule** -- a PyTorch Lightning DataModule for streaming WebDataset files

- **BioNeMo NIMs**: Easy-to-use, enterprise-ready _inference_ microservices with built-in API endpoints. NIMs are engineered for scalable, self- or cloud-hosted deployment of optimized, production-grade biomolecular foundation models. Check out the growing list of BioNeMo NIMs [here](https://build.nvidia.com/explore/biology).

Use the **recipes** and **sub-packages** when you need to train, fine-tune, or customize models. Use **NIMs** when you need production-ready inference against existing models.

Get notified of new releases, bug fixes, critical security updates, and more for biopharma. [Subscribe.](https://www.nvidia.com/en-us/clara/biopharma/product-updates/)

## BioNeMo User Success Stories

[Enhancing Biologics Discovery and Development With Generative AI](https://www.nvidia.com/en-us/case-studies/amgen-biologics-discovery-and-development/) - Amgen leverages BioNeMo and DGX Cloud to train large language models (LLMs) on proprietary protein sequence data, predicting protein properties and designing biologics with enhanced capabilities. By using BioNeMo, Amgen achieved faster training and up to 100X faster post-training analysis, accelerating the drug discovery process.

[Cognizant to apply generative AI to enhance drug discovery for pharmaceutical clients with NVIDIA BioNeMo](https://investors.cognizant.com/news-and-events/news/news-details/2024/Cognizant-to-apply-generative-AI-to-enhance-drug-discovery-for-pharmaceutical-clients-with-NVIDIA-BioNeMo/default.aspx) - Cognizant leverages BioNeMo to enhance drug discovery for pharmaceutical clients using generative AI technology. This collaboration enables researchers to rapidly analyze vast datasets, predict interactions between drug compounds, and create new development pathways, aiming to improve productivity, reduce costs, and accelerate the development of life-saving treatments.

[Cadence and NVIDIA Unveil Groundbreaking Generative AI and Accelerated Compute-Driven Innovations](https://www.cadence.com/en_US/home/company/newsroom/press-releases/pr/2024/cadence-and-nvidia-unveil-groundbreaking-generative-ai-and.html) - Cadence's Orion molecular design platform will integrate with BioNeMo generative AI tool to accelerate therapeutic design and shorten time to trusted results in drug discovery. The combined platform will enable pharmaceutical companies to quickly generate and assess design hypotheses across various therapeutic modalities using on-demand GPU access.

Find more user stories on NVIDIA's [Customer Stories](https://www.nvidia.com/en-us/case-studies/?industries=Healthcare%20%26%20Life%20Sciences&page=1) and [Technical Blog](https://developer.nvidia.com/blog/search-posts/?q=bionemo) sites.
