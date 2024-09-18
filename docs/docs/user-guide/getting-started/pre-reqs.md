# Hardware and Software Prerequisites

Before you begin using the BioNeMo framework, please ensure the following prerequisites are met.

## Hardware

The BioNeMo Framework is compatible with environments that have access to NVIDIA GPUs. Bfloat16 precision requires an Ampere generation GPU or higher. Tested GPUs include: H100, A100, and RTX A6000. There is mixed support for GPUs without bfloat16 support, such as V100, T4, Quadro RTX 8000, and GeForce RTX 2080 Ti. GPUs with known issues include: Tesla K80.

## Software

The BioNeMo Framework is supported on x86 Linux systems. Please use the project’s Docker images to develop and execute the code.

Please ensure that the following are installed in your desired execution environment:
* Appropriate GPU drivers
* Docker (with GPU support, Docker Engine 19.03 or above)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to allow Docker to access the GPUs
