# Base image with apex and transformer engine, but without NeMo or Megatron-LM.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
FROM ${BASE_IMAGE} AS bionemo2-base

# Install NeMo dependencies.
WORKDIR /build

ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# See NeMo readme for the latest tested versions of these libraries
ARG APEX_COMMIT=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout ${APEX_COMMIT} && \
  pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir \
  --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

# Transformer Engine pre-1.7.0. 1.7 standardizes the meaning of bits in the attention mask to match
ARG TE_COMMIT=7d576ed25266a17a7b651f2c12e8498f67e0baea
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git fetch origin ${TE_COMMIT} && \
  git checkout FETCH_HEAD && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

# Install core apt packages.
RUN apt-get update \
  && apt-get install -y \
  libsndfile1 \
  ffmpeg \
  git \
  curl \
  pre-commit \
  sudo \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y gnupg

# Check the nemo dependency for causal conv1d and make sure this checkout
# tag matches. If not, update the tag in the following line.
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.0.post2

# Mamba dependancy installation
RUN pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/state-spaces/mamba.git@v2.0.3

RUN pip install hatchling   # needed to install nemo-run
ARG NEMU_RUN_TAG=34259bd3e752fef94045a9a019e4aaf62bd11ce2
RUN pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@${NEMU_RUN_TAG}

FROM bionemo2-base AS pip-requirements

# Copy and install pypi depedencies.
RUN mkdir /tmp/pip-tmp
WORKDIR /tmp/pip-tmp

COPY requirements-dev.txt requirements-test.txt requirements-cve.txt /tmp/pip-tmp/

# We want to only copy the requirements.txt, setup.py, and pyproject.toml files for *ALL* sub-packages
# but we **can't** do COPY sub-packages/**/{requirements.txt,...} /<destination> because this will overwrite!
# So....we copy everything into a temporary image and remove everything else!
# Later, we can copy the result from the temporary image and get what we want
# **WITHOUT** invalidating the cache for successive layers!
COPY sub-packages/ /tmp/pip-tmp/sub-packages
# remove all directories that aren't the top-level sub-packages/bionemo-{xyz}
RUN find sub-packages/ -type d | grep "bionemo-[a-zA-Z0-9\-]*/" | xargs rm -rf && \
    # only keep the requirements-related files
    find sub-packages/ -type f | grep -v -E "requirements.txt|pyproject.toml|setup.py" | xargs rm

FROM bionemo2-base AS dev

RUN mkdir -p /workspace/bionemo2/
WORKDIR /workspace/bionemo2

# We get the sub-packcages/ top-level structure + requirements.txt files
COPY --from=pip-requirements /tmp/pip-tmp/ /workspace/bionemo2/

RUN pip install -r requirements-dev.txt -r requirements-test.txt -r requirements-cve.txt

# We calculate paths to each requirements.txt file and dynamically construct the pip install command.
# This command will expand to something like:
#   pip install --disable-pip-version-check --no-cache-dir \
#      -r bionemo-core/requirements.txt \
#      -r bionemo-pytorch/requirements.txt \
#      -r bionemo-lmm/requirements.txt \
#      (etc.)
RUN X=""; for sub in $(echo sub-packages/bionemo-*); do X="-r ${sub}/requirements.txt ${X}"; done; eval "pip install --disable-pip-version-check --no-cache-dir ${X}"

# Delete the temporary /build directory.
WORKDIR /workspace
RUN rm -rf /build

# Addressing Security Scan Vulnerabilities
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx
RUN apt-get update  && \
    apt-get install -y openssh-client=1:8.9p1-3ubuntu0.10 && \
    rm -rf /var/lib/apt/lists/*
RUN apt purge -y libslurm37 libpmi2-0 && \
    apt autoremove -y
RUN source /usr/local/nvm/nvm.sh && \
    NODE_VER=$(nvm current) && \
    nvm deactivate && \
    nvm uninstall $NODE_VER && \
    sed -i "/NVM/d" /root/.bashrc && \
    sed -i "/nvm.sh/d" /etc/bash.bashrc

# Create a non-root user to use inside a devcontainer.
ARG USERNAME=bionemo
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

RUN find /usr/local/lib/python3.10/dist-packages/ -type f -print0 | xargs -0 -P 0 -n 10000 chown $USERNAME:$USER_GID

ENV PATH="/home/bionemo/.local/bin:${PATH}"


# Create a release image with bionemo2 installed.
FROM dev AS release

# Install 3rd-party deps
COPY ./3rdparty /build
WORKDIR /build/Megatron-LM
RUN pip install --disable-pip-version-check --no-cache-dir .

WORKDIR /build/NeMo
RUN pip install --disable-pip-version-check --no-cache-dir .[all]
WORKDIR /workspace
RUN rm -rf /build

# Install bionemo2 submodules
WORKDIR /workspace/bionemo2/
COPY VERSION .
COPY ./sub-packages /workspace/bionemo2/sub-packages
# Dynamically install the code for each bionemo namespace package.
RUN for sub in sub-packages/bionemo-*; do pushd ${sub} && pip install --no-build-isolation --no-cache-dir --disable-pip-version-check --no-deps -e . && popd; done

WORKDIR /workspace/bionemo2/
COPY ./scripts ./scripts
COPY ./README.md ./
