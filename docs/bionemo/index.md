# What is BioNeMo?

Generative AI and large language models (LLMs) are achieving incredible breakthroughs in chemistry and biology, such as enabling 3D protein structure prediction, property prediction, and even the generation of novel protein sequences and molecules. This progress has facilitated developments in the pharmaceutical industry, such as antibody design, small-molecule drug design, and newer approaches like RNA aptamer and peptide-based therapeutics. As each of these pieces comes into play, their respective models may need additional fine-tuning or optimization to thoroughly explore or understand the biomolecular space, leading to the need for centralized infrastructure for model development and deployment.

**BioNeMo Framework** is a free to use collection of programming tools and packages offering access to optimized, pre-trained biomolecular models and workflows, along with versatile functionalities for building and customizing models, including training and fine-tuning. Capabilities span various workloads and therapeutic modalities, such as molecular generation and representation learning, protein structure prediction and representation learning, protein-ligand and protein-protein docking, and DNA/RNA/single-cell embedding.

**BioNeMo NIMs** are easy-to-use enterprise-ready inference microservices with built-in API endpoints. NIMs are engineered for scalable, self-hosted or cloud-hosted deployment of optimized, production-grade biomolecular foundation models on any cloud or data center. Check out the growing list of BioNeMo NIMs [here](https://build.nvidia.com/explore/biology).

When selecting between the BioNeMo Framework and BioNeMo NIMs, the specific requirements of your project should guide your decision. The BioNeMo Framework is suitable for scenarios that necessitate model training, fine-tuning, or customization, as it provides a comprehensive suite of tools and packages for building and tailoring models. However, this framework requires a more complex setup procedure and familiarity with the code base. In contrast, BioNeMo NIMs are optimized for inference-only workflows, offering easy-to-use, enterprise-ready inference microservices with built-in API endpoints that are quick to deploy and utilize. Additionally, NIMs may be optimized for specific hardware, potentially resulting in accelerated inference times and increased throughput. As a general guideline, the BioNeMo Framework is recommended for use cases that necessitate custom model development or require a high degree of control over the modeling process. Conversely, BioNeMo NIMs are the preferred choice when only inference against an existing model is required.

![](./images/bionemo_overview_2.png)

## BioNeMo Customer Success Stories

[Enhancing Biologics Discovery and Development With Generative AI](https://www.nvidia.com/en-us/case-studies/amgen-biologics-discovery-and-development/) - Amgen is utilizing NVIDIA's BioNeMo service and DGX Cloud to train large language models (LLMs) on proprietary protein sequence data, enabling the prediction of protein properties and the design of biologics with enhanced properties. By leveraging BioNeMo's turnkey solution, Amgen was able to rapidly train and fine-tune protein LLMs, achieving faster training and up to 100X faster post-training analysis compared to open-source options. This partnership accelerates the drug discovery process by enabling the generation of hypotheses about biologic effects, properties, and side effects, and facilitates the development of more effective and targeted treatments.

[NVIDIA, Evozyne Create Generative AI Model for Proteins](https://blogs.nvidia.com/blog/generative-ai-proteins-evozyne/) - Evozyne is using a pretrained AI model included in BioNeMo to create proteins with potential applications in healthcare and clean energy. The researchers also used BioNeMo to train their own model, ProT-VAE, which combines a transformer model with a variational autoencoder, allowing them to generate millions of protein sequences in just a few seconds and explore new proteins with useful functions.

[Accelerating Drug Discovery at Receptor.AI with NVIDIA BioNeMo Cloud APIs](https://developer.nvidia.com/blog/accelerating-drug-discovery-at-receptor-ai-with-nvidia-bionemo-cloud-apis/) - Receptor.AI integrated NVIDIA's BioNeMo with their computer-assisted drug discovery (CADD) platform, achieving significant performance improvements and cost savings by shifting tasks like virtual screening and ligand pose prediction from CPU-based processing to accelerated computing on NVIDIA's platform. The integration enabled Receptor.AI to identify hit compounds with higher predicted affinity to the target protein FADS1, with a 49% reduction in costs per instance-hour. The collaboration will further develop AI-based drug discovery platforms for biotech and pharmaceutical companies, leveraging NVIDIA's hardware and software accelerations.

[NVIDIA BioNeMo Expands Computer-Aided Drug Discovery With New Foundation Models](https://blogs.nvidia.com/blog/bionemo-ai-drug-discovery-foundation-models-microservices/) - Companies are leveraging BioNeMo to accelerate molecular simulations and utilize large language models for drug discovery applications. These advancements include integrating BioNeMo with existing platforms, such as Cadence's Orion, to speed up molecular simulation, and developing pipelines for therapeutic assets using AI-accelerated workflows, as with Insilico Medicine. Additionally, companies like Iambic and Recursion are contributing predictive models, such as NeuralPLexer and Phenom-Beta, to BioNeMo to better understand protein structure changes and cell function in response to drug molecules.

[Cognizant to apply generative AI to enhance drug discovery for pharmaceutical clients with NVIDIA BioNeMo](https://investors.cognizant.com/news-and-events/news/news-details/2024/Cognizant-to-apply-generative-AI-to-enhance-drug-discovery-for-pharmaceutical-clients-with-NVIDIA-BioNeMo/default.aspx) - Cognizant is utilizing NVIDIA's BioNeMo platform to leverage generative AI technology in drug discovery, enabling researchers to rapidly analyze vast datasets, predict interactions between drug compounds, and create new development pathways. This collaboration aims to provide clients with a suite of model-making services, including pretrained models and APIs, to accelerate drug discovery with reduced manual intervention and infrastructure maintenance. The goal is to improve productivity, reduce costs, and increase the speed of bringing life-saving treatments to market.

[Cadence and NVIDIA Unveil Groundbreaking Generative AI and Accelerated Compute-Driven Innovations](https://www.cadence.com/en_US/home/company/newsroom/press-releases/pr/2024/cadence-and-nvidia-unveil-groundbreaking-generative-ai-and.html) - Cadence's Orion molecular design platform will be integrated with NVIDIA's generative AI tool, BioNeMo, to accelerate therapeutic design capabilities and shorten time to trusted results in drug discovery. The combined platform will enable pharmaceutical companies to quickly and reliably generate and assess design hypotheses across various therapeutic modalities, including biologics, peptides, and small molecules, using on-demand GPU access. BioNeMo models will provide structure prediction, small molecule generation, and molecular property prediction, which can be further profiled and iteratively enhanced and designed with Orion tools.

Find more customer stories on NVIDIA's [Customer Stories](https://www.nvidia.com/en-us/case-studies/?industries=Healthcare%20%26%20Life%20Sciences&page=1) and [Technical Blog](https://developer.nvidia.com/blog/search-posts/?q=bionemo) pages.

## BioNeMo Framework: Fundamentals
BioNeMo Framework provides versatile functionalities for developing and training large-scale biology-based models. BioNeMo allows users to build and train biomolecular models by providing access to pre-trained models and common model components for accelerating drug discovery workflows. Built for supercomputing scale, the framework allows developers to easily configure and train distributed multi-node jobs with minimal code.

![](./images/bionemo_overview_1.png)

BioNeMo is built on [NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html), a scalable and cloud-native generative AI framework for researchers to create, customize, and deploy large language models (LLMs). NeMo provides a robust environment for working with large learning models, including [NVIDIA Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/megatron.html) models. The BioNeMo Framework provides enhancements to PyTorch Lighting, such as hyperparameter configurability with YAML files and checkpoint management. Users can conveniently and quickly train models using these features, test them for desired tasks, and integrate them alongside existing applications.

Some of the key features of BioNeMo Framework are:

* Development and training of large transformer models using NVIDIA's Megatron framework.
* Easy to configure multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision.
* Model training recipes that can be readily implemented on DGX compute infrastructure.
* Logging with Tensorboard and Weights and Biases to monitor the model training process.


## BioNeMo Framework: Available Models
State-of-the-art models are being continually integrated into the BioNeMo Framework. The available catalog consists of: 1) Models developed by NVIDIA, 2) Models contributed by NVIDIA’s ecosystem partners, and 3) Community models further enhanced by NVIDIA to take advantage of GPU acceleration. The BioNeMo Framework currently offers the following pre-trained models:

| **Model**                                               | **Modality**             | **Uses**                                      |
|---------------------------------------------------------|--------------------------|-----------------------------------------------|
| [MegaMolBART](./models/megamolbart.md)                  | Small Molecule           | Representation Learning + Molecule Generation |
| [MolMIM](./models/molmim.md)                            | Small Molecule           | Representation Learning + Molecule Generation |
| [ESM-1nv](./models/esm1-nv.md)                          | Protein                  | Representation Learning                       |
| [ESM-2nv 650M](./models/esm2-nv.md)                     | Protein                  | Representation Learning                       |
| [ESM-2nv 3B](./models/esm2-nv.md)                       | Protein                  | Representation Learning                       |
| [EquiDock DIPS Model](./models/equidock.md)             | Protein                  | Protein-Protein Complex Formation             |
| [EquiDock DB5 Model](./models/equidock.md)              | Protein                  | Protein-Protein Complex Formation             |
| [OpenFold](./models/openfold.md)                        | Protein                  | Protein Structure Prediction                  |
| [ProtT5nv](./models/prott5nv.md)                        | Protein                  | Representation Learning                       |
| [DiffDock Confidence Model](./models/diffdock.md)       | Protein + Molecule       | Generation of Ligand Poses                    |
| [DiffDock Score Model](./models/diffdock.md)            | Protein + Molecule       | Generation of Ligand Poses                    |
| [DNABERT](./models/dnabert.md)                          | DNA                      | Representation Learning                       |
| [Geneformer](./models/geneformer.md)                    | Single Cell              | Representation Learning                       |

When building your workflow, take the following steps to choose the right model:

1. Consider the model size and assess your compute availability, as larger models demand significant computational resources.
2. Examine your data type, as different models are designed for specific data types such as DNA, RNA, protein, small molecule, or mixed data.
3. Define the target outcome of your workflow, whether it's generation, property prediction, or other tasks.

Use the detailed information in each Model Card, which provides insights into the training data, compute requirements, capabilities, and optimizations of each model. By reviewing these Model Cards carefully, you can select a model that aligns with your workflow, ultimately maximizing the effectiveness and efficiency of your research.
