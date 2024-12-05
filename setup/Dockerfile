# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: check if the following missing dependencies are needed
#   - lcms2=2.12=hddcbb42_0
#   - libgcc-ng=11.1.0=hc902ee8_8
#   - libstdcxx-ng=11.1.0=h56837e0_8
#   - pycairo=1.20.1=py38hf61ee4a_0

# Most of the first part is directly copied from NeMo v1.23 Dockerfile https://github.com/NVIDIA/NeMo/blob/v1.23.0/Dockerfile
# kept all dependencies necessary for nemo.collections.nlp, nemo.core, nemo.utils, nemo.collections.common

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
ARG DGL_IMAGE=gitlab-master.nvidia.com:5005/clara-discovery/bionemo:dgl_builder_v1.1.4
FROM ${DGL_IMAGE} AS dgl
# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as base

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
  libfreetype6 \
  swig && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

# need this otherwise apex or transformer engine crashes since it runs out of host memory
# c++: fatal error: Killed signal terminated program cc1plus
ARG MAX_JOBS=2
ENV MAX_JOBS=${MAX_JOBS}

# Install megatron core, this can be removed once 0.3 pip package is released
# We leave it here in case we need to work off of a specific commit in main,
# see commit from https://github.com/NVIDIA/NeMo/blob/d2283e3620cd7f99dbe29fdf079757ab9f6cdf01/Dockerfile#L67
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && \
  cd Megatron-LM && \
  git checkout 240a8ef7a21df201e47b5b2ae33cc5f4c5486849 && \
  pip install .

# Distributed Adam support for multiple dtypes
# see commit from https://github.com/NVIDIA/NeMo/blob/d2283e3620cd7f99dbe29fdf079757ab9f6cdf01/Dockerfile#L73
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout f058162b215791b15507bb542f22ccfde49c872d && \
  pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

# Transformer Engine used in NeMo v1.22 dockerfile
# see commit from https://github.com/NVIDIA/NeMo/blob/d2283e3620cd7f99dbe29fdf079757ab9f6cdf01/Dockerfile#L79
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git fetch origin da30634a6c9ccdbb6c587b6c93b1860e4b038204 && \
  git checkout FETCH_HEAD && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=0 MPI_HOME=/usr/local/mpi pip install .

WORKDIR /tmp/

# uninstall stuff from base container
RUN pip uninstall -y sacrebleu torchtext

WORKDIR /tmp/nemo
# install flash attention
RUN pip install flash-attn
# install numba for latest containers
# ImportError: Numba needs NumPy 1.24 or less
RUN pip install numba>=0.57.1
ARG NEMO_VERSION=1.23.0
# Check that NEMO_VERSION is set. Build will fail without this. Expose NEMO and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "$NEMO_VERSION" && \
  /bin/echo "export NEMO_VERSION=${NEMO_VERSION}" >> /root/.bashrc && \
  /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc

# Install NeMo
RUN git clone -b v$NEMO_VERSION https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    pip uninstall -y nemo_toolkit sacrebleu && \
    pip install -r requirements/requirements_lightning.txt && \
    pip install -r requirements/requirements_common.txt && \
    sed -i "/torch/d" requirements/requirements.txt && \
    sed -i "/triton/d" requirements/requirements.txt && \
    sed -i "/megatron_core/d" requirements/requirements_nlp.txt && \
    sed -i "/nltk/d" requirements/requirements_nlp.txt && \
    pip install -r requirements/requirements.txt && \
    pip install -r requirements/requirements_nlp.txt && \
    pip install ".[nlp]"


# need this for backward compatibility: pytests rely on configs in /workspace/nemo/examples
RUN mkdir -p /workspace/nemo/
RUN cp -r /tmp/nemo/NeMo/scripts /workspace/nemo/scripts
RUN cp -r /tmp/nemo/NeMo/examples /workspace/nemo/examples
RUN cp -r /tmp/nemo/NeMo/tests /workspace/nemo/tests
RUN cp -r /tmp/nemo/NeMo/tutorials /workspace/nemo/tutorials

ENV TERM=xterm
ENV FORCE_CUDA=1

# makes the interactive bash shell "login" have this in the prompt
RUN echo 'export PS1="(bionemo docker) $PS1"' > /docker_bashrc

WORKDIR /workspace/
# TODO: Long term, docker should not be running as root.
ENV HOME=/root
# copy built DGL and install it
# dgl_builder is created with setup/Dockerfile.dgl_builder
COPY --from=dgl /dgl ./dgl
RUN cd dgl/python && python setup.py install && cd ../.. && rm -rf dgl

# Please populate the labels below if using a custom base container (ie not NeMo GA)
# For nightly NeMo builds, only com.nvidia.nemo.description should be provided
# (release and git_hash labels are filled automatically).
# LABEL "com.nvidia.nemo.release"="r1.19.0"
# LABEL "com.nvidia.nemo.git_hash"="a5fbsegy"
# LABEL "com.nvidia.nemo.description"="Cusotom build with XYZ fixed"

## Installing system dependencies and tools
##
## NOTE: Need to resolve issue with libboost-all-dev for vina py. dependency.
##      If we apt install libboost-dev-all, then, on import torch, we observe an error
##      stating that ucm_set_global_opts is undefined in libucs. The error stack will
##      always look something like the following:
##
## .........................................: in <module>
##     import torch
## /usr/local/lib/python3.10/dist-packages/torch/__init__.py:234: in <module>
##     from torch._C import *  # noqa: F403
## E   ImportError: /opt/hpcx/ucx/lib/libucs.so.0: undefined s
##

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3-setuptools python3-dev libboost-all-dev swig \
        git unzip tmux vim emacs-nox tree xterm \
        liblmdb-dev libopenbabel-dev python3-openbabel openbabel libopenbabel7

## libboost-all-dev is required by vina==1.2.5 for its compilation
## While at the runtime only boost-thread and boost-filesystem are needed
RUN pip install --no-build-isolation vina==1.2.5
RUN apt purge *mpi* -y \
    && apt autoremove -y \
    && apt install libboost-thread-dev libboost-filesystem-dev -y \
    && rm -rf /var/lib/apt/lists/*

# special handling for openbabel:
# apt installs the openbabel3 library to /usr/include/openbabel3, but the openbabel python package
# assumes that it is under /usr/local/include/openbabel3. The error will look like this on install:
#       Error: SWIG failed. Is Open Babel installed?
#       You may need to manually specify the location of Open Babel include and library directories. For example:
#         python setup.py build_ext -I/usr/local/include/openbabel3 -L/usr/local/lib
#         python setup.py install
#       [end of output]
# To fix, we symlink!
RUN ln -s /usr/include/openbabel3 /usr/local/include/openbabel3
# NOTE: openbabel is inlcuded in setup/requirements.txt

# Optional: Install code-server to enable easy remote development on a container
# More info about code-server be found here: https://coder.com/docs/code-server/v4.4.0
# ADD https://github.com/coder/code-server/releases/download/v4.4.0/code-server_4.4.0_amd64.deb code-server_4.4.0_amd64.deb
# RUN dpkg -i ./code-server_4.4.0_amd64.deb && rm -f code-server_4.4.0_amd64.deb
#
# # Install extensions from the marketplace
# RUN code-server --install-extension ms-python.python
# moved -test and -dev up in build to resolve build errors
# COPY files first **BEFORE** installing from them => use the layer cache effectively!
COPY setup/requirements*.txt /tmp/
COPY setup/install_lmdb.sh /tmp/install_lmdb.sh
# installing PyG from a seperate requirements file
RUN pip install --no-cache-dir -r /tmp/requirements-pyg.txt
RUN --mount=type=secret,id=GITLAB_TOKEN GITLAB_TOKEN=$(cat /run/secrets/GITLAB_TOKEN) pip install -r /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements-test.txt

# FIXME [mgreaves] unknown why installing fw2nim doesn't work
#    39 35.05 ERROR: Could not find a version that satisfies the requirement fw2nim==0.1.0 (from versions: none)
#    39 35.05 ERROR: No matching distribution found for fw2nim==0.1.0
#    39 ERROR: executor failed running [/bin/sh -c GITLAB_TOKEN=$(cat /run/secrets/GITLAB_TOKEN) pip install -r /tmp/requirements.txt]: exit code: 1
# but infra-bionemo does:
#    37 3.282 Collecting infra-bionemo==0.1.0 (from -r /tmp/requirements-dev.txt (line 7))
#    37 3.366   Downloading https://gitlab-master.nvidia.com/api/v4/projects/118589/packages/pypi/files/487d2342614e85a010143523f44fcee5058ff71b90b5e10a0d4a0ce259f66230/infra_bionemo-0.1.0-py3-none-any.whl (29 kB)
# NOTE that doing a local docker run w/ personal API token **does work**, it's the gitlab RO API token that does not work
# THIS IS ONLY A GITLAB CI PROBLEM !!!
RUN --mount=type=secret,id=GITLAB_TOKEN GITLAB_TOKEN=$(cat /run/secrets/GITLAB_TOKEN) pip install -r /tmp/requirements.txt
# need special install for lmdb: it cannot be installed via pip, but be built from source
# but unfortunately, https://github.com/jnwatson/py-lmdb.git isn't PEP-512 compliant, so we have to manually run setup.py
RUN /tmp/install_lmdb.sh
# requirements-cve.txt *must* be installed last to patch security vulns in pip packages
RUN pip install -r /tmp/requirements-cve.txt


# AWS CLI is inside requirements.txt. Add config file.
# This config file does not contain access keys.
RUN mkdir -p $HOME/.aws
COPY setup/aws-config-swiftstack $HOME/.aws/config

# Recompile Megatron helpers -- NOTE this may need to be done before running job later
RUN NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])') && \
   cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron && \
   make

#########################################################################################
# Apply NeMo patches for key upstream changes between releases
#  TODO whenever we upgrade nemo see which can be removed.
#########################################################################################
# 1. Setting the workdir to the base nemo directory for all nemo patches

# NOTE: This is to support the MegaMolBart LM model compatability with Jupyter notebooks and ipdb.
#       This is exposed through the `iteractive=True` constructor parameter.
#       In MegaMolBart, specifically, this patch adds support for recongifure_minibatch as a flag
#       in its decode method.
# TODO [mgreaves] Upstream nemo-v-1.23-support_interactive_megamolbart_lm-decode_reconfigure_minibatch patch and
# NOTE: hiddens_support_train_v_inf_outputs.patch applies the patch from https://github.com/NVIDIA/NeMo/pull/8466 which fixes an issue
#        where training uses the same hiddens key for inference (eg always the gaussian noised z rather than z_mean).
# NOTE: data_sampler_constant_len.patch.patch applies the patch from https://github.com/NVIDIA/NeMo/pull/8576 which fixes an issue
#        where you restart a run after an epoch.
# NOTE: nemo-lr-scheduler.patch adds patch for WarmupAnnealDecayHold NeMo LR Scheduler
COPY ./setup/*.patch ./
RUN NEMO_INST_DIR=$(python -c 'import nemo; from pathlib import Path; print(Path(nemo.__path__[0]).parent)') && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/nemo-v-1.23-support_interactive_megamolbart_lm-decode_reconfigure_minibatch.patch && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/hiddens_support_train_v_inf_outputs.patch && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/data_sampler_constant_len.patch && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/nemo-lr-scheduler.patch && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/nemo-v-1.23-import_except_torch_amp_grad_scaler.patch && \
    rm *.patch
###################################################################################
# Special Handling for Security Vulnerabilities
###################################################################################

# Remove nvm as it's not expected users will need this, and it introduces unfixable vulnerabilties:
# https://nvd.nist.gov/vuln/detail/CVE-2023-32002
RUN /bin/bash -c "source /usr/local/nvm/nvm.sh && nvm deactivate && nvm unload && sed -i '/NVM/d' /root/.bashrc" \
    && rm -rf /usr/local/nvm \
    && /bin/bash -c "sed -i '/NVM/d' /root/.bashrc && sed -i '/nvm.sh/d' /etc/bash.bashrc"

# Remove OpenCV: We do not use it and it adds a security vulnerability (fix: https://github.com/opencv/opencv_contrib/pull/3480)
# https://nvd.nist.gov/vuln/detail/CVE-2023-2617
# Remove fonttool: We do not use it and it adds a security vulnerability: https://github.com/advisories/GHSA-6673-4983-2vx5
RUN pip uninstall -y opencv-python opencv fonttools \
    && rm -rf \
       /usr/local/lib/python3.10/dist-packages/cv2 \
       /usr/local/include/opencv4 \
       /usr/local/share/opencv4 \
       /usr/local/lib/cmake/opencv4 \
       /usr/local/bin/opencv*

# Remove CVEs:
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx \
  && DEBIAN_FRONTEND=noninteractive apt remove -y libslurm37 libpmi2-0 openssh-client \
  && DEBIAN_FRONTEND=noninteractive apt autoremove -y

# Remove lightning-app. Nemo and BioNeMo doesn't use it, and it introduces security vulnerabilities:
# https://github.com/advisories/GHSA-cgwc-qvrx-rf7f
# https://github.com/advisories/GHSA-mr7h-w2qc-ffc2
RUN rm -rf /usr/local/lib/python3.10/dist-packages/lightning/app

####
#### NOTE: THIS MUST BE THE ABSOLUTE **LAST** STEP BEFORE ENTRYPOINT !!!!!
####       SPECIFICALLY, THE SECTION LABELED "BioNeMo Build"
####       DO NOT MAKE OTHER LAYERS **AFTER** THIS SECTON!
####       OTHERWISE IT WILL MAKE REBUILDS ON SOURCE CHANGES TAKE LONGER THAN NECESSARY BY INVALIDATING THE LAYER CACHE.
####

#########################################################################################
# BioNeMo Build
#########################################################################################

# directory permissioning commands:
#   - openfold training requires the installation of libraries via
#   ./examples/protein/openfold/scripts/install_third_party.sh,
#   at container run time, rather than image build time.
#   - the installers require the user submitting the install commands
#   to have access to certain locations, where typically root has access
#   but the user does not.
RUN chmod a+rwx /usr/local /usr/local/lib /usr/local/include /usr/local/bin
RUN mkdir -p /usr/local/scripts /usr/local/lib/cmake/
RUN chmod a+rwx /usr/local/scripts /usr/local/lib/cmake/


### The location of the installed library in the container. Set BIONEMO_HOME env to change this for development purposes.
ARG BIONEMO_HOME=/workspace/bionemo
ENV BIONEMO_HOME=${BIONEMO_HOME}
RUN mkdir -p ${BIONEMO_HOME}
WORKDIR ${BIONEMO_HOME}
### Copy the bionemo repo contents for distribution
COPY LICENSE README.md artifact_paths.yaml download_artifacts.py launch.sh setup.py pyproject.toml ${BIONEMO_HOME}/
COPY setup ${BIONEMO_HOME}/setup
COPY hydra_plugins ${BIONEMO_HOME}/hydra_plugins
COPY tokenizers ${BIONEMO_HOME}/tokenizers
# copy the package contents *last* as most churn occurs here
# --> we can cache the lower-chrun things in previous layers
COPY examples ${BIONEMO_HOME}/examples
COPY tests ${BIONEMO_HOME}/tests
COPY bionemo ${BIONEMO_HOME}/bionemo
COPY ci/scripts ${BIONEMO_HOME}/ci/scripts
COPY docs ${BIONEMO_HOME}/docs

### install bionemo
WORKDIR ${BIONEMO_HOME}
# we already install dependencies earlier, directly from the requirements*.txt files
RUN pip install --no-deps -e .

# FIXME: If BIONEMO_HOME _is not_ /workspace/bionemo, then this ENTRYPOINT is invalid!
ENTRYPOINT ["/workspace/bionemo/setup/startup.sh"]
