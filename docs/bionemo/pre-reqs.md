# Hardware and Software Prerequisites

Before you begin using the BioNeMo framework, please ensure the following prerequisites are met.

## Hardware

The BioNeMo Framework is compatible with environments that have access to NVIDIA GPUs. `bfloat16` precision requires an Ampere generation GPU or higher. There is mixed support for GPUs without `bfloat16`.

### GPU Support Matrix

| GPU | Support |
|------|---------|
| H100 | Full |
| A100 | Full |
| RTX A6000 | Full |
| V100 | Partial |
| T4 | Partial |
| Quadro RTX 8000 | Partial |
| GeForce RTX 2080 Ti | Partial |
| Tesla K80 | Known Issues |

## Software

The BioNeMo Framework is supported on x86 Linux systems.

Please ensure that the following are installed in your desired execution environment:
* Appropriate GPU drivers (minimum version: 535)
* Docker (with GPU support, Docker Engine 19.03 or above)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to allow Docker to access the GPUs
