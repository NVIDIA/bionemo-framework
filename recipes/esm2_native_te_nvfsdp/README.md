# ESM-2 training with nvFSDP and custom pytorch training loop

To build the docker image with gitlab authentication, first ensure you have a gitlab user access
token stored in your `~/.netrc` file with the following format:

```
machine gitlab-master.nvidia.com
  login user
  password <your_gitlab_token>
```

Then, build the docker image with the following command:

```bash
docker build --secret id=netrc,src=$HOME/.netrc -t my_image .
```

This will make sure the netrc file is available to the docker build process for authentication.

We can remove this once nvFSDP is publically available on github.com.

## Running training

Run training with

```bash
docker run --rm -it --gpus all my_image torchrun train.py --config-name L0_sanity
```

# Lepton script

```bash
#!/bin/bash

# Download the environment setup script from Lepton's GitHub repository, make it executable, and source it to initialize the environment variables.
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

# This doesn't seem to work from lepton nodes
curl --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" https://gitlab-master.nvidia.com/clara-discovery/bionemo-recipes/-/archive/main/bionemo-recipes-main.tar.gz\?path\=recipes/esm2_native_te_nvfsdp --output bionemo-recipes-main.tar.gz

tar xzf bionemo-recipes-main.tar.gz
cd bionemo-recipes-main-recipes-esm2_native_te_nvfsdp/recipes/esm2_native_te_nvfsdp/

PIP_CONSTRAINT= pip install -r requirements.txt

torchrun \
    --rdzv_id ${LEPTON_JOB_NAME} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --nproc-per-node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node-rank ${NODE_RANK} \
    train.py --config-name L1_15B_perf_test
```
