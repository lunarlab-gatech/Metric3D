# Plain Ubuntu base (not an nvidia/cuda or pytorch/pytorch image) so this environment
# doesn't depend on upstream base images that get deprecated/deleted over time.
# GPU access comes from the NVIDIA driver on the host via the container runtime
# (--gpus all); the CUDA runtime libraries needed at run time are bundled in the
# torch/torchvision wheels installed below.
FROM ubuntu:22.04

# Set default user arguments which should be overwritten by docker build
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

# System packages needed to build/run mmcv, OpenCV, matplotlib, and to install torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        openssh-client \
        ninja-build \
        wget \
        ca-certificates \
        sudo \
        tmux \
        python3 \
        python3-pip \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Pin torch/torchvision to the exact CUDA build required by requirements_v2.txt
# (torch==2.0.1, torchvision==0.15.2), so this environment stays reproducible even
# after the upstream pytorch/pytorch and nvidia/cuda base images are deprecated.
RUN pip install --no-cache-dir \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# Create a non-root user with matching UID/GID
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    groupadd --gid 110 render && \
    usermod -aG render ${USERNAME}

# Add a custom .bashrc with warm colors and fixed hostname
RUN echo 'export PS1="\\[\\e[1;31m\\]\\u@metric3d_container\\[\\e[0m\\]:\\[\\e[1;33m\\]\\w\\[\\e[0m\\]\\$ "' > /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

# Set default user & shell
USER ${USERNAME}
WORKDIR /home/${USERNAME}
SHELL ["/bin/bash", "-c"]
RUN echo "export USER=${USERNAME}" >> ~/.bashrc

RUN echo "set -g mouse on" >> /home/${USERNAME}/.tmux.conf
