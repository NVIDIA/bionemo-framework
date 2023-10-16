# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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


ARG BASE_IMAGE=nvcr.io/nvidia/nemo:23.06
FROM ${BASE_IMAGE}

WORKDIR /workspace/

# copy built DGL and install it
# dgl_builder is created with setup/Dockerfile.dgl_builder
COPY --from=gitlab-master.nvidia.com:5005/clara-discovery/bionemo:dgl_builder /dgl ./dgl
RUN cd dgl/python && python setup.py install && cd ../.. && rm -rf dgl

# Please populate the labels below if using a custom base container (ie not NeMo GA)
# For nightly NeMo builds, only com.nvidia.nemo.description should be provided
# (release and git_hash labels are filled automatically).
# LABEL "com.nvidia.nemo.release"="r1.19.0"
# LABEL "com.nvidia.nemo.git_hash"="a5fbsegy"
# LABEL "com.nvidia.nemo.description"="Cusotom build with XYZ fixed"

# Activating base environment
ENV TERM=xterm

## Installing dependencies and tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3-setuptools python3-dev swig \
        wget git unzip tmux vim tree xterm && \
    rm -rf /var/lib/apt/lists/*

COPY setup/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
COPY setup/requirements-linter.txt /tmp/requirements-linter.txt
RUN pip3 install -r /tmp/requirements-linter.txt


## Install NGC cli in container for artifact download
ENV NGC_CLI_INSTALL_PREFIX=/usr/local/bin
RUN wget -O /tmp/ngccli_linux.zip --content-disposition \
    "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.30.1/files/ngccli_linux.zip" && \
    unzip /tmp/ngccli_linux.zip -d ${NGC_CLI_INSTALL_PREFIX} && \
    chmod u+x ${NGC_CLI_INSTALL_PREFIX}/ngc-cli/ngc && \
    ln -fns  ${NGC_CLI_INSTALL_PREFIX}/ngc-cli/ngc ${NGC_CLI_INSTALL_PREFIX}/ngc && \
    rm /tmp/ngccli_linux.zip

# Recompile Megatron helpers -- NOTE this may need to be done before running job later
RUN NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])') && \
   cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron && \
   make

#########################################################################################
# BioNeMo Build
#########################################################################################

# The location of the installed library in the container. Set BIONEMO_HOME env to change this for development purposes.
ARG BIONEMO_HOME=/opt/nvidia/bionemo
ENV BIONEMO_HOME=${BIONEMO_HOME}

# Copy the repo
COPY LICENSE README.md artifact_paths download_models.sh launch.sh pytest.ini ${BIONEMO_HOME}/
COPY bionemo ${BIONEMO_HOME}/bionemo
COPY docs ${BIONEMO_HOME}/docs
COPY examples ${BIONEMO_HOME}/examples
COPY generated ${BIONEMO_HOME}/generated
COPY hydra_plugins ${BIONEMO_HOME}/hydra_plugins
COPY proto ${BIONEMO_HOME}/proto
COPY setup ${BIONEMO_HOME}/setup
COPY tests ${BIONEMO_HOME}/tests

# NEMO_HOME must be unset for default NeMo install in /usr/local/lib/python3.8/dist-packages/nemo
ARG NEMO_HOME
ENV NEMO_HOME=${NEMO_HOME}

ENV PYTHONPATH=${BIONEMO_HOME}:${BIONEMO_HOME}/generated:${NEMO_HOME}:$PYTHONPATH

COPY ./setup/startup.sh /opt/nvidia
COPY ./tokenizers /tokenizers

RUN mkdir -p /workspace/bionemo
RUN ln -s /opt/nvidia/bionemo/examples/ /workspace/bionemo/examples
RUN echo 'export PS1="(bionemo docker) $PS1"' > /docker_bashrc

WORKDIR /workspace

ENTRYPOINT ["/opt/nvidia/startup.sh"]
