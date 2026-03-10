# To build a production image:
# docker build -t <image_name> --target production .
#
# To build a development image:
# docker build -t <image_name> --target development --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) .

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${FROM_IMAGE_NAME} AS base

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace/

# Copy requirements first for better layer caching
COPY ./requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt \
    && apt-get update \
    && apt-get install -y libsndfile1 ffmpeg \
    && pip install -U flash-attn


RUN pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils

# ----------------- Production Stage -----------------
FROM base AS production

WORKDIR /workspace/
COPY . .
# Add a CMD for production if you have a main script to run
# e.g., CMD ["python", "app.py"]


# ----------------- Development Stage -----------------
FROM base AS development

WORKDIR /workspace/

# Install development specific packages
RUN apt-get update && apt-get install -y htop

# Create a non-root user for development
ARG USERNAME
ARG USER_UID
ARG USER_GID

# Ensure build arguments are provided for development builds
RUN if [ -z "$USERNAME" ] || [ -z "$USER_UID" ] || [ -z "$USER_GID" ]; then \
        echo "Error: For development builds, you must provide USERNAME, USER_UID, and USER_GID build arguments." >&2; \
        exit 1; \
    fi

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -l --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to the non-root user
USER $USERNAME
