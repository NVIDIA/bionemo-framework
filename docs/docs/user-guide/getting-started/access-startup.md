# Access and Startup

The BioNeMo Framework is free to use and easily accessible. The preferred method of accessing the software is through
the BioNeMo Docker container, which provides a seamless and hassle-free way to develop and execute code. By using the
Docker container, you can bypass the complexity of handling dependencies, ensuring that you have a consistent and
reproducible environment for your projects.

In this section of the documentation, we will guide you through the process of pulling the BioNeMo Docker container and
setting up a local development environment. By following these steps, you will be able to quickly get started with the
BioNeMo Framework and begin exploring its features and capabilities.

## Startup Instructions

BioNeMo is compatible with a wide variety of computing environments, including both local workstations, data centers,
and Cloud Service Providers (CSPs) such as Amazon Web Services, Microsoft Azure, Google Cloud Platform, and Oracle Cloud
Infrastructure, and NVIDIA’s own DGX Cloud. Note that exact configuration details may differ based on your preferred
setup, but the instructions included below should provide a consistent starting point across environments.

### Running the Container on a Local Machine

This section will provide instructions for running the BioNeMo Framework container on a local workstation. This process
will involve the following steps:

1. Pulling the container from the NGC registry
2. Running a Jupyter Lab instance inside the container for local development

#### Pull Docker Container from NGC

You now pull the BioNeMo Framework container using the following command:

```bash
docker pull {{ docker_url }}:{{ docker_tag }}
```

#### Run the BioNeMo Framework Container

Now that you have pulled the BioNeMo Framework container, you can run it as you would a normal Docker container. For
instance, to get basic shell access you can run the following command:

```bash
docker run --rm -it --gpus all \
  {{ docker_url }}:{{ docker_tag }} \
  /bin/bash
```

Because BioNeMo is distributed as a Docker container, standard arguments can be passed to the `docker run` command to
alter the behavior of the container and its interactions with the host system. For more information on these arguments,
refer to the [Docker documentation](https://docs.docker.com/reference/cli/docker/container/run/).

In the next section, [Initialization Guide](./initialization-guide.md), we will present some useful `docker run` command
variants for common workflows.

## Running on Any Major CSP with the NVIDIA GPU-Optimized VMI

The BioNeMo Framework container is supported on cloud-based GPU instances through the
**NVIDIA GPU-Optimized Virtual Machine Image (VMI)**, available for
[AWS](https://aws.amazon.com/marketplace/pp/prodview-7ikjtg3um26wq#pdp-pricing),
[GCP](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi),
[Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11?tab=overview), and
[OCI](https://cloudmarketplace.oracle.com/marketplace/en_US/listing/165104541).
NVIDIA VMIs are built on Ubuntu and provide a standardized operating system environment across cloud infrastructure for
running NVIDIA GPU-accelerated software. These images are pre-configured with software dependencies such as NVIDIA GPU
drivers, Docker, and the NVIDIA Container Toolkit. More details about NVIDIA VMIs can be found in the
[NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nvidia_vmi).

The general steps for launching the BioNeMo Framework container using a CSP are:

1. Launch a GPU-equipped instance running the NVIDIA GPU-Optimized VMI on your preferred CSP. Follow the instructions for
    launching a GPU-equipped instance provided by your CSP.
2. Connect to the running instance using SSH and run the BioNeMo Framework container exactly as outlined in the
    [Running the Container on a Local Machine](#running-the-container-on-a-local-machine) section on
    the Access and Startup page.
