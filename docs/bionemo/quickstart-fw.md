# Quickstart Guide

The purpose of this quickstart is to make users familiar with the different components, features and functionalities of the BioNeMo framework. It outlines how to access various resources related to BioNeMo, what is provided inside the BioNeMo container, and how a user can adapt different parts of BioNeMo for their intended use case. 

The former part of this guide contains information about 
- configuring BioNeMo for data processing
- training of BioNeMo models
- information about various output files generated

The latter section includes documentation relevant to code development with BioNeMo. 
  
If you’re new to BioNeMo, the best way to get started is to take a look at the following [tutorials](./tutorials-fw.md):

- [Custom dataset pre-processing](./notebooks/custom-dataset-preprocessing-fw.ipynb)
- [BioNeMo LLM training](./notebooks/model_training_esm1nv.ipynb) 
- [Fine-tuning](./notebooks/bionemo-finetuning-overview.ipynb) 
- [Inferencing](./notebooks/MMB_GenerativeAI_Inference_with_examples.ipynb) 

Finally, some advanced steps related to using BioNeMo on NVIDIA's Base Command Platform (BCP) are also provided. Refer to [Running BCP jobs](./bcp-specific-commands-fw.md). 


## Setup
### Accessing BioNeMo Framework 

To get the access to BioNeMo, visit the {{bionemo_info_url}}.

### NGC Setup

NVIDIA NGC is the cloud platform offering fully managed services for AI solutions. Learn more about [NGC here](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html). 

A user is required to configure the NGC access to obtain the BioNeMo container, relevant example datasets, pre-trained models, and more. It is also needed for using the NGC compute resources / DGX Cloud for training LLM and performing inference at scale. 

To configure the NGC cli and setup an API token:

1. Sign into [NGC](https://catalog.ngc.nvidia.com/).
2. The NGC CLI must be installed on the workstation, and it can be downloaded from [NGC setup](https://ngc.nvidia.com/setup).
3. An NGC API token can be acquired at the [NGC setup](https://ngc.nvidia.com/setup) page when logged in with NVIDIA credentials. The API key is only displayed once for security, so it should be copied to a safe location. 
4. Follow the instructions provided to [setup NGC Access](https://ngc.nvidia.com/setup/api-key).


### [Optional] Weights and Biases Setup

BioNeMo provides a feature for logging the LLM training progress with Weights and Biases. It requires an account and a [Weights and Biases API key](https://docs.wandb.ai/guides/track/public-api-guide). The API key should be setup on the workstation or computer from which the job is launched as described on the [Weights and Biases website](https://docs.wandb.ai/guides/track/public-api-guide#authentication).

---

:::{note}
The following sections on [Docker Container Access](./quickstart-fw.md#docker-container-access), and [Code Access](./quickstart-fw.md#code-access) are intended for users who wish to run BioNeMo tasks on a local workstation. For running BioNeMo on DGX-nodes using BCP, please refer to relevant [tutorials](./tutorials-fw.md). 
:::


### Docker Container Access

The latest Docker container should be downloaded from NGC once the NGC API token is acquired. To log in to the NGC container registry:

```bash
docker login {deploy_ngc_registry}
Username: $oauthtoken
Password <insert NGC API token here>
```

Details related to the BioNeMo container addreess on NGC, organization ID, Team, (and ACE in case of DGX-Cloud) will be provided to the users in NGC invitation e-mail. To pull the container, execute the following, where  `{deploy_ngc_org}` and `{deploy_ngc_team}` should be updated with the correct NGC org and team names, respectively: 

```bash
docker pull {deploy_ngc_registry}/{deploy_ngc_org_team}/{deploy_container_name}:{deploy_container_tag}
```

Alternatively, once the `launch.sh` script is configured (refer to [Configuring the Launch Script](#configuring-the-launch-script)) the container can be pulled with `bash launch.sh pull`.

### Code Access

BioNeMo code is provided inside the container. The following bash script will launch the docker container and copy the code directory from the container to a workstation path:

```bash
CONTAINER="{deploy_ngc_registry}/{deploy_ngc_org_team}/{deploy_container_name}:{deploy_container_tag}"
DEST_PATH="."
CONTAINER_NAME=bionemo
docker run --name $CONTAINER_NAME -itd --rm $CONTAINER bash
docker cp $CONTAINER_NAME:/opt/nvidia/bionemo $DEST_PATH
docker kill $CONTAINER_NAME
```

### Configuring the Launch Script

If the user has access to the BioNeMo code and needs to download and run the container, the `launch.sh` script is provided for convenience located in the root of the code directory. This script can automate several tasks, including container pulling, pushing, and launching. Alternatively, the container can also be pulled from the NGC registry, as mentioned above.

The launch script requires a settings file called `.env`, which will be automatically created in the project directory if it does not exist on first launch. If created manually, it should be named `.env` and placed inside the repo directory on the workstation. All of the variables are described in the `usage` section of the launch script.

Some of the variables are required to be updated according to the BioNeMo framework and compute access. `BIONEMO_IMAGE` is required, and should point to the BioNeMo container address on NGC. Some of the `.env` variables are optional, for example, `WANDB_API_KEY` is only required if the logging is monitored using Weights and Biases (WandB). Similarly, `NGC_CLI_TEAM` is optional and can be omitted if it does not exist. In contrast, `NGC_CLI_API_KEY` is essential for downloading the BioNeMo container image, pre-trained model weights, relevant datasets, and using NGC DGX-cloud compute resources. A list of variables are provided in the table here. 


| Variable      | Description |
| ----------- | ----------- |
|  `BIONEMO_IMAGE`     | Required to obtain BioNeMo container       |
|  `WANDB_API_KEY`  | Optional, to enable monitoring via Weights and Biases       |
|  `NGC_CLI_API_KEY`     | Required to access BioNeMo resources, obtain container, etc.    |
|  `NGC_CLI_ORG`  | Required to access ORG specific resources, including containers.       |
|  `ACE_VALUE`     | Required for ACE specific compute resource allocations   |
|  `NGC_CLI_TEAM`  | Optional, for team specific resource logistics |



In the sample configuration below, update `{deploy_ngc_org}` and `{deploy_ngc_team}` with the correct NGC org and team name, respectively.  

```bash
BIONEMO_IMAGE={deploy_ngc_registry}/{deploy_ngc_org_team}/{deploy_container_name}:{deploy_container_tag}               # Container with tag
BIONEMO_PATH=/workspace/bionemo                                          # Location of code to be used /workspace/bionemo or /opt/nvidia/bionemo
PROJECT_PATH=$(pwd)                                                      # Path of env config and optional development code
DATA_PATH=${HOME}/data                                                   # Local path to save downloaded and processed data
RESULT_PATH=${HOME}/result/bionemo_experiments                           # Path for training results
WANDB_API_KEY=<Insert WANDB API KEY>                                     # WandB logging requires API key configuration
JUPYTER_PORT=8888                                                        # Jupyter port for inference notebooks
PROJECT_MOUNT=/workspace/bionemo                                         # Location of library in container /workspace/bionemo for dev work or /opt/nvidia/bionemo for non-dev use                                      
REGISTRY={deploy_ngc_registry}                                                         # Only required for registry login
REGISTRY_USER='$oauthtoken'                                              # Only required for registry login
NGC_CLI_API_KEY=$(grep -m 1 apikey ~/.ngc/config | head -n 1 | cut -d' ' -f3) # Requires NGC cli configuration
NGC_CLI_ORG={deploy_ngc_org}
NGC_CLI_TEAM={deploy_ngc_team}
NGC_CLI_FORMAT_TYPE=ascii
```

In order to prevent downloading and processing of data each time a new container is launched, be sure to set a valid path for `DATA_PATH` in the `.env` file. This directory will be mounted inside the container at `/data`. Once a container has been pulled and the `.env` file has been setup, the container can be started in interactive development mode with `bash launch.sh dev`. 



## Training Configuration

:::{note}
The following describes how to configure a **single node** training on a workstation to test and configure a sample pre-training job. Training at scale should be performed on BCP, as described in the [Tutorials section](./tutorials-fw.md).
:::

### Python Job Script

A Python-based pre-training script imports the YAML parameter files, and can be found in the corresponding subdirectory of `examples`, for example, `examples/molecule/megamolbart/scripts/pretrain.py`. These scripts run the data processing and training jobs and typically will not need to to be changed.

### Command Line Configuration

BioNeMo uses YAML based parameter files for training and model configuration. The existing configuration files can be found in the corresponding `conf` subdirectory of `examples`, for example, `examples/molecule/megamolbart/conf`. Additional files can be created to suit other configurations. Refer to the [NeMo Fundamentals Tutorial](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) for an overview of the YAML configuration files.

Job execution parameters can be changed in the YAML configuration files or on the command line. By changing these values, it is possible to configure the size of the training run (number of nodes and GPUs), model precision, and so on. Multi-node training has been validated on BCP-based systems.

Every BioNeMo YAML base config file has the following configuration structure:
- `trainer`
- `exp_manager`
- `model`

Some commonly changed configuration parameters are:

```yaml
trainer:
  devices: 8 # number of GPUs per node
  num_nodes: 8 # number of nodes

model:
  micro_batch_size: 8 # batch size for each model instance
```

Refer to the [Command Line Configuration section](./bionemo-fw-for-model-training-fw.md#command-line-configuration) for more information.

### Running a Sample Job

For convenience, the job execution commands have been wrapped into a series of shell scripts that demonstrate functionality by model and are located in the corresponding `scripts` directory within `examples`. These scripts are suitable for demonstration and testing a single-node training on a workstation. For multi-node training, refer to the [Tutorials](./tutorials-fw.md) and [Running BioNeMo on DGX-Cloud with BCP](./bcp-specific-commands-fw.md) section.

- MegaMolBART: `examples/molecule/megamolbart/scripts/pretrain_quick.sh`
- ESM-1nv: `examples/protein/esm1nv/scripts/pretrain_quick.sh`
- ProtT5nv: `examples/protein/prott5nv/scripts/pretrain_quick.sh`

These scripts first require some variables to be set in their `CONFIG` section. BioNeMo uses [Hydra](https://github.com/facebookresearch/hydra) for configuring a job, which allows arguments to be changed using the `=` operator, `+` to add, and `~` to remove the argument from the CLI. Important script configuration flags are:

- `-c`, `--config`: Path to a YAML config file. The default is specified in each of the respective scripts.
- `-f`, `--data-files`: The set or subset of processed data files (in CSV format) to use for training. Consecutive ranges of files can be supplied -- for three files (x000.csv, x001.csv, and x002.csv), the range can be supplied as "x[000..002]". However, the addition of the required shell escapes to this can become complicated, so it's advisable to use `_OP_` instead of `[` and `_CL_` instead of `]`, making the range `x_OP_000..002_CL_`
- `-o`, `--result_dir`: Set the path for the results.
- `-a`, `--args`: Additional arguments passed to the training. Use `--args` flag followed by a string of additional arguments, for example, to use tensor model parallelism of 2 with 4 GPUs: `--args "++trainer.devices=4 ++model.tensor_model_parallel_size=2"`

The two command options are `preprocess`, which will download and process the data, and `train` which performs pre-training using the data. An example is provided for executing one of the scripts once the `CONFIG` section has been edited:

```bash
cd examples/protein/esm1nv/scripts && \
bash pretrain_quick.sh \
--config esm-1nv_config \
--data-files x_OP_000..010_CL_ \
train
```

For more advanced customization, the shell script can be modified with additional training arguments. 

### Pre-Processing ZINC15 Data

For preprocessing MegaMolBART data as a demo, a large amount (~100GB) of data in the form of multiple files must be downloaded for the selected ZINC15 tranches. To test the preprocessing and training on a subset of the files, change the filename in the `links_file` setting in the config file (`examples/molecule/megamolbart/conf/pretrain_base.yaml`) from `ZINC-downloader.txt` to `ZINC-downloader-sample.txt`, for example:

```yaml
links_file: '${oc.env:BIONEMO_HOME}/examples/molecule/megamolbart/dataset/ZINC-downloader-sample.txt' # to process a subset of the ZINC15 data
```

## Training Output

### Log Files

BioNeMo creates a number of log files during training in the results directory:

- `cmd-args.log` : the Python commands and any additional parameter overrides used for training
- `hparams.yaml` : the final set of all hyperparameters used for training
- `git-info.log` : the commit hash of the code used for training and the results of `git diff` on this code to store any additional modifications from this commit
- `nemo_log_globalrank-0_localrank-0.txt` : BioNeMo creates a number of logs, depending on how many GPUs are used in training. Local rank is the rank order within a node and global rank is the rank order across all nodes.
- `nemo_error_log.txt` : error logs from NeMo
- `lightning_logs.txt` : PyTorch Lightning logs

### Checkpoints

Checkpoints are stored in the `checkpoints` directory and are managed by NeMo. YAML configuration file contains multiple checkpoint related parameters in the `exp_manager` section. For example, it is possible to ensure only the checkpoints with top k validation loss values are saved. Refer to the [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/core/exp_manager.html#) for more information. The latest checkpoint is automatically reloaded from the `checkpoints` directory when resuming training.

NeMo can also create a `*.nemo` checkpoint which has the optimizer states removed and can be used for inference or fine-tuning tasks. More information about restoring a pre-trained model from `*.nemo` checkpoint can be found [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/results.html).

### Tensorboard and Weights and Biases Files

BioNeMo supports optional logging with Tensorboard and Weights and Biases. Tensorboard logging will create the standard `events.out.tfevents.....` file, while Weights and Biases logs are stored in a directory called `wandb`. Weights and Biases logging can be done offline if cluster restrictions do not allow upload of files during training. To enable offline logging and upload the contents after training, follow the directions for [offline sync](https://docs.wandb.ai/guides/technical-faq/setup#can-i-run-wandb-offline).

## Inference 

Trained BioNeMo models are provided on NGC for use. These models can be loaded with a gRPC interface provided in BioNeMo. Example notebooks are provided in the corresponding `nbs` directory in `examples`, for example, `examples/molecule/megamolbart/nbs`. 

If a local copy of the code exists on the workstation, download the pre-trained models with `bash launch.sh download`. Alternatively, the models can be downloaded inside the container by running `source download_models.sh && download_bionemo_models`. Model weights downloaded from NGC will require an [NGC API key](https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-key) to be configured.

To launch the gRPC interface and a Jupyter notebook service in the corresponding directory, ensure that `JUPYTER_PORT` is set correctly in the `.env` file, then run `bash launch.sh dev -c <model_name>`, where `<model_name>` is one of the following: `megamolbart`, `prott5nv`, `esm-1nv`. Then open a browser at `http://<<HOST_NAME>>:$JUPYTER_PORT` to execute the notebook.

## Code Development Setup
### Code Organization

The BioNeMo repo is organized by biological entity (`molecule`, `protein`) and by deep learning model, for example, `molecule/megamolbart` or `protein/esm1nv`. Shared code can be found within utils files or directories. These are the most important directories and their purpose:

* `bionemo`: The library location, including functionality for data processing, data loading, models, and associated functions.
* `examples`: The files required to run a training or inferences: Python training scripts, YAML configuration files, BCP/shell scripts, and Jupyter notebooks.
* `models`: The location for trained models that are downloaded.
* `tokenizers`: The location for data domain specific tokenizers
* `setup`: Dockerfile and associated setup scripts.


### Mounting Code in a Container

It is possible to mount an alternative version of BioNeMo code inside the container for development purposes. This requires setting `PROJECT_PATH` in the `.env` file to the path of the code to be mounted, and then launching the container in interactive development mode with `bash launch.sh dev`. Inside the container, the `PROJECT_PATH` environment variable will be defined and set to the container path (`/workspace/bionemo`). 

It may also be necessary to recompile the Megatron helpers, which can be done with the script `setup/recompile_megatron_helper.sh`. This recompilation should also be done immediately before training starts on clusters as a best practice.

### Build and Push Container

It is highly suggested to pull a prebuilt container for BioNeMo development. However, if a container must be built, the `launch.sh` script can also be used to build and push containers to a registry. Execute `bash launch.sh build -p` to build a container. If pushing to a registry is desired, `bash launch.sh push` will complete this task.
