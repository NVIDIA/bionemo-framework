# What is BioNeMo?


<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __User Guide__

    ---

    Install BioNeMo and set up your environment to start accelerating your bioinformatics workflows

    [:octicons-arrow-right-24: User Guide](user-guide/getting-started)

-   :material-scale-balance:{ .lg .middle } __API Reference__

    ---

    Access comprehensive documentation of BioNeMo's functions and classes

    [:octicons-arrow-right-24: API Reference](API_reference/bionemo/core/api/)

-   :material-format-font:{ .lg .middle } __Models__

    ---

    Explore detailed instructions and best practices for using BioNeMo in your research

    [:octicons-arrow-right-24: Models](models)

-   :fontawesome-brands-markdown:{ .lg .middle } __Datasets__

    ---

    Join the BioNeMo community and learn how to contribute to this open-source project

    [:octicons-arrow-right-24: Datasets](datasets)

</div>


Generative AI and large language models (LLMs) are achieving incredible breakthroughs in chemistry and biology, such as enabling 3D protein structure prediction, property prediction, and even the generation of novel protein sequences and molecules. This progress has facilitated developments in the pharmaceutical industry, such as antibody design, small-molecule drug design, and newer approaches like RNA aptamer and peptide-based therapeutics. As each of these pieces comes into play, their respective models may need additional fine-tuning or optimization to thoroughly explore or understand the biomolecular space, leading to the need for centralized infrastructure for model development and deployment.

**BioNeMo Framework** is a free to use collection of programming tools and packages offering access to optimized, pre-trained biomolecular models and workflows, along with versatile functionalities for building and customizing models, including training and fine-tuning. Capabilities span various workloads and therapeutic modalities, such as molecular generation and representation learning, protein structure prediction and representation learning, protein-ligand and protein-protein docking, and DNA/RNA/single-cell embedding.

**BioNeMo NIMs** are easy-to-use enterprise-ready inference microservices with built-in API endpoints. NIMs are engineered for scalable, self-hosted or cloud-hosted deployment of optimized, production-grade biomolecular foundation models on any cloud or data center. Check out the growing list of BioNeMo NIMs [here](https://build.nvidia.com/explore/biology).

![](assets/old_images/bionemo_overview_2.png)

## BioNeMo Framework: Fundamentals

BioNeMo Framework provides versatile functionalities for developing and training large-scale biology-based models. BioNeMo allows users to build and train biomolecular models by providing access to pre-trained models and common model components for accelerating drug discovery workflows. Built for supercomputing scale, the framework allows developers to easily configure and train distributed multi-node jobs with minimal code.

![](assets/old_images/bionemo_overview_1.png)

BioNeMo is built on [NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html), a scalable and cloud-native generative AI framework for researchers to create, customize, and deploy large language models (LLMs). NeMo provides a robust environment for working with large learning models, including [NVIDIA Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/megatron.html) models. The BioNeMo Framework provides enhancements to PyTorch Lighting, such as hyperparameter configurability with YAML files and checkpoint management. Users can conveniently and quickly train models using these features, test them for desired tasks, and integrate them alongside existing applications.

Some of the key features of BioNeMo Framework are:

- Development and training of large transformer models using NVIDIA's Megatron framework.
- Easy to configure multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision.
- Model training recipes that can be readily implemented on DGX compute infrastructure.
- Logging with Tensorboard and Weights and Biases to monitor the model training process.
