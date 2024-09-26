# What is BioNeMo?

BioNeMo is a software ecosystem produced by NVIDIA for the development and deployment of deep learning models in life sciences. It provides a set of tools to help researchers build, train, and deploy AI models for various biological applications. The main components of BioNeMo are:

- **BioNeMo Framework**: a free-to-use collection of programming tools and packages offering access to optimized, pre-trained biomolecular models and workflows, along with versatile functionalities for building and customizing models, including training and fine-tuning. Capabilities span various workloads and therapeutic modalities, such as molecular generation and representation learning, protein structure prediction and representation learning, protein-ligand and protein-protein docking, and DNA/RNA/single-cell embedding.

- **BioNeMo NIMs**: easy-to-use, enterprise-ready _inference_ microservices with built-in API endpoints. NIMs are engineered for scalable, self- or cloud-hosted deployment of optimized, production-grade biomolecular foundation models. Check out the growing list of BioNeMo NIMs [here](https://build.nvidia.com/explore/biology).

When choosing between the BioNeMo Framework and BioNeMo NIMs, consider your project's specific requirements. The Framework is ideal for scenarios that require model training, fine-tuning, or customization, offering a comprehensive suite of tools and packages. In contrast, NIMs are optimized for inference-only workflows, providing easy-to-use, enterprise-ready microservices with built-in API endpoints. As a rule, use the Framework for custom model development or high-control modeling, and NIMs for inference against existing models.

## BioNeMo User Success Stories

[Enhancing Biologics Discovery and Development With Generative AI](https://www.nvidia.com/en-us/case-studies/amgen-biologics-discovery-and-development/) - Amgen leverages BioNeMo and DGX Cloud to train large language models (LLMs) on proprietary protein sequence data, predicting protein properties and designing biologics with enhanced capabilities. By using BioNeMo, Amgen achieved faster training and up to 100X faster post-training analysis, accelerating the drug discovery process.

[Cognizant to apply generative AI to enhance drug discovery for pharmaceutical clients with NVIDIA BioNeMo](https://investors.cognizant.com/news-and-events/news/news-details/2024/Cognizant-to-apply-generative-AI-to-enhance-drug-discovery-for-pharmaceutical-clients-with-NVIDIA-BioNeMo/default.aspx) - Cognizant leverages BioNeMo to enhance drug discovery for pharmaceutical clients using generative AI technology. This collaboration enables researchers to rapidly analyze vast datasets, predict interactions between drug compounds, and create new development pathways, aiming to improve productivity, reduce costs, and accelerate the development of life-saving treatments.

[Cadence and NVIDIA Unveil Groundbreaking Generative AI and Accelerated Compute-Driven Innovations](https://www.cadence.com/en_US/home/company/newsroom/press-releases/pr/2024/cadence-and-nvidia-unveil-groundbreaking-generative-ai-and.html) - Cadence's Orion molecular design platform will integrate with BioNeMo generative AI tool to accelerate therapeutic design and shorten time to trusted results in drug discovery. The combined platform will enable pharmaceutical companies to quickly generate and assess design hypotheses across various therapeutic modalities using on-demand GPU access.

Find more user stories on NVIDIA's [Customer Stories](https://www.nvidia.com/en-us/case-studies/?industries=Healthcare%20%26%20Life%20Sciences&page=1) and [Technical Blog](https://developer.nvidia.com/blog/search-posts/?q=bionemo) sites.

## BioNeMo Framework: Fundamentals

BioNeMo Framework provides versatile functionalities for developing and training large-scale biology-based models. BioNeMo allows users to build and train biomolecular models by providing access to pre-trained models and common model components for accelerating drug discovery workflows. Built for supercomputing scale, the framework allows developers to easily configure and train distributed multi-node jobs with minimal code.

BioNeMo is built on [NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html), a scalable and cloud-native generative AI framework for researchers to create, customize, and deploy large language models (LLMs). NeMo provides a robust environment for working with large learning models, including [NVIDIA Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/megatron.html) models. The BioNeMo Framework provides enhancements to PyTorch Lighting, such as hyperparameter configurability with YAML files and checkpoint management. Users can conveniently and quickly train models using these features, test them for desired tasks, and integrate them alongside existing applications.

Some of the key features of BioNeMo Framework are:

- Development and training of large transformer models using NVIDIA's Megatron framework.
- Easy to configure multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision.
- Model training recipes that can be readily implemented on DGX compute infrastructure.
- Logging with Tensorboard and Weights and Biases to monitor the model training process.
