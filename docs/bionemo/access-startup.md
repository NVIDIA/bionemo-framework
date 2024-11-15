# Access and Startup

The BioNeMo Framework is free to use and easily accessible. Users can pull the BioNeMo Framework Docker container to develop and execute code. Below, we outline the steps to access the latest version.

An open-source version of the BioNeMo Framework is coming soon and will be available on GitHub.

# Access the BioNeMo Framework

## NGC Account and API Key Configuration

NVIDIA GPU Cloud (NGC) is a portal of enterprise services, software, and support for AI and HPC workloads. The [NGC Catalog](https://catalog.ngc.nvidia.com/) is a collection of GPU-accelerated software, models and containers that speed up end-to-end AI workflows. The BioNeMo Framework container is available on NGC.

1. Create a free account on [NGC](https://ngc.nvidia.com/signin) and log in.
2. At the top right, click on the **User > Setup > Generate API Key**, then click **+ Generate API Key** and **Confirm**. Copy and store your API Key in a secure location.

You can now view the BioNeMo Framework container [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) or by searching the [NGC Catalog](https://catalog.ngc.nvidia.com/) for “BioNeMo Framework”. Feel free to explore the resources available to you in the Catalog.

# Startup Instructions

Now that you can access the BioNeMo Framework container, it is time to get up and running. BioNeMo is compatible across a variety of computing environments, keeping in mind users with local workstations and data centers, users of major CSPs (e.g., AWS, Azure, GCP, and OCI), and users of NVIDIA’s DGX Cloud infrastructure.

## Running the Container on a Local Machine

### Pull Docker Container from NGC

Within the [NGC Catalog](https://catalog.ngc.nvidia.com/), navigate to **BioNeMo Framework > Tags > Get Container**, and copy the image path for the latest tag.

Open a command prompt on your machine and enter the following:

```bash
docker login nvcr.io

    Username: $oauthtoken
    Password: <YOUR_API_KEY>
```

You can now pull the container with the command below. Replace the <TAG> string with your desired version tag available [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework/tags):

```bash
BIONEMO_IMAGE_PATH=nvcr.io/nvidia/clara/bionemo-framework:<TAG>
docker pull $BIONEMO_IMAGE_PATH
```

To automatically retrieve the latest container, you can use the following commands:

```bash
LATEST_TAG=$(ngc registry image list --format_type=csv --column=tag nvidia/clara/bionemo-framework | grep "1\." | head -1 | cut -f 2 -d',')

BIONEMO_IMAGE_PATH=nvcr.io/nvidia/clara/bionemo-framework:${LATEST_TAG}

docker pull $BIONEMO_IMAGE_PATH
```

### Run Docker Container

First, create a local workspace directory (to be mounted to the home directory of the Docker container to persist data). You can then launch the container. We recommend running the container in a JupyterLab environment, as per the below command:

```bash
docker run --rm -d --gpus all -p 8888:8888 \
  -v <YOUR_WORKSPACE>:/workspace/bionemo/<YOUR_WORKSPACE> $BIONEMO_IMAGE_PATH \
  "jupyter lab --allow-root --ip=* --port=8888 --no-browser \
  --NotebookApp.token='' --NotebookApp.allow_origin='*' \
  --ContentsManager.allow_hidden=True --notebook-dir=/workspace/bionemo"
```

Explanation:
* **Docker**: The first line runs a Docker container in detached mode (`-d`), uses all available GPUs for the container (``--gpus all``), and maps it to port 8888.
* **Volume Mapping**: Maps host directory into the home directory of the container.
* **JupyterLab Command**: Customizable command line which allows root access (``--allow-root``), binding to all IP addresses on the specified port (``--ip=* --port=8888``), disables browser launch (``--no-browser``) and token authentication requirements (``--NotebookApp.token``), shows hidden files by setting (``.allow_hidden=True``), and sets the starting working directory to ``/workspace/bionemo``.

## Running the Container in the Cloud through Major CSPs

### Launch Instance Through NVIDIA VMI

The BioNeMo Framework container is supported on cloud-based GPU instances through the **NVIDIA GPU-Optimized Virtual Machine Image (VMI)**, available for [AWS](https://aws.amazon.com/marketplace/pp/prodview-7ikjtg3um26wq#pdp-pricing), [GCP](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi), [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11?tab=overview), and [OCI](https://cloudmarketplace.oracle.com/marketplace/en_US/listing/165104541). NVIDIA VMIs are built on Ubuntu and provide a standardized operating system environment across clouds for running NVIDIA GPU-accelerated software. They are pre-configured with software dependencies such as NVIDIA GPU drivers, Docker, and the NVIDIA Container Toolkit. More details about NVIDIA VMIs can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nvidia_vmi).

The general steps below should be adapted according to the CSP:
1. Launch a GPU instance running the NVIDIA GPU-Optimized VMI (e.g. AWS EC2).
2. Connect to the running instance, and then pull and run the BioNeMo Framework container exactly as outlined in the [Running the Container on a Local Machine](#running-the-container-on-a-local-machine) section above.

### Integration with Cloud Services

BioNeMo is compatible with various cloud services. Check out blogs about BioNeMo on [SageMaker](https://aws.amazon.com/blogs/industries/find-the-next-blockbuster-with-nvidia-bionemo-framework-on-amazon-sagemaker/) (example code [repository](https://github.com/aws-samples/amazon-sagemaker-with-nvidia-bionemo)), [ParallelCluster](https://aws.amazon.com/blogs/hpc/protein-language-model-training-with-nvidia-bionemo-framework-on-aws-parallelcluster/) (example code [repository](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/14.bionemo)), and [EKS](https://aws.amazon.com/blogs/hpc/accelerate-drug-discovery-with-nvidia-bionemo-framework-on-amazon-eks/) (example code [repository](https://github.com/awslabs/data-on-eks/tree/main/ai-ml/bionemo)).

## Running the Container on DGX Cloud

For DGX Cloud users, NVIDIA Base Command Platform (BCP) includes a central user interface with managed compute resources. It can be used to manage datasets, workspaces, jobs, and users within an organization and team. This creates a convenient hub for monitoring job execution, viewing metrics and logs, and monitoring resource utilization. NVIDIA DGX Cloud is powered by Base Command Platform. More information can be found on the [BCP website](https://docs.nvidia.com/base-command-platform/index.html).

### NGC CLI Configuration

NVIDIA NGC Command Line Interface (CLI) is a command-line tool for managing Docker containers in NGC. You can download it on your local machine as per the instructions [here](https://org.ngc.nvidia.com/setup/installers/cli).

Once installed, run `ngc config set` to establish NGC credentials:
* **API key**: Enter your API Key
* **CLI output**: Accept the default (ascii format) by pressing `Enter`
* **org**: Choose from the list which org you have access to
* **team**: Choose the team you are assigned to
* **ace**: Choose an ACE, otherwise press `Enter` to continue

Note that the **org** and **team** are only relevant when pulling private containers/datasets from NGC created by you or your team. For BioNeMo Framework, use the default value.

You can learn more about NGC CLI installation [here](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#installing-ngc-cli). Note that the NGC documentation also discusses how to mount your own [datasets](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#managing-datasets) and [workspaces](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#managing-workspaces).

### Running the BioNeMo Framework Container

On your local machine, run the following command to launch your job, ensuring to replace the relevant fields with your settings:

```bash
ngc batch run \
	--name <YOUR_JOB_NAME> \
	--team <YOUR_TEAM> \
	--ace <YOUR_ACE> \
	--instance dgxa100.80g.1.norm \
	--image <BIONEMO_IMAGE_PATH> \
	--port 8888 \
	--workspace <YOUR_WORKSPACE>:/workspace/bionemo/<YOUR_WORKSPACE>:RW \
	--datasetid <YOUR_DATASET> \
	--result /result \
	--total-runtime 1D \
	--order 1 \
	--label <YOUR_LABEL> \
	--commandline "jupyter lab --allow-root --ip=* --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' --ContentsManager.allow_hidden=True --notebook-dir=/workspace/bionemo & sleep infinity"
```

Explanation:
* `--name`: Name of your job
* `--team`: Team that you are assigned in NGC org
* `--ace`: ACE that you are assigned
* `--instance`: GPU instance type for the job (e.g. `dgxa100.80g.1.norm` for single-GPU A100 instance)
* `--image`: BioNeMo Framework container image
* `--port`: Port number to access JupyterLab
* `--workspace`: Optional (Mount NGC workspace to container with read/write access to persist data)
* `--datasetid`: Optional (Mount dataset to container)
* `--result`: Directory to store job results
* `--order`: Order of the job
* `--label`: Job label, allowing quick filtering on NGC dashboard
* `--commandline`: Command to run inside the container, in this case, starting JupyterLab and keeping it running with `sleep infinity`

To launch your Jupyter notebook in the browser, click on your [job in the NGC Web UI](https://bc.ngc.nvidia.com/jobs) and then click the URL under the Service Mapped Ports. You may also set up a Remote Tunnel to access a running job to execute and edit your code using VS Code locally or via the browser, as discussed [here](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#setting-up-and-accessing-visual-studio-code-via-remote-tunnel).
