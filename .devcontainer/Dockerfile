FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        ccache \
        cmake \
        curl \
        g++ \
        gcc \
        gdb \
        git \
        less \
        libfftw3-dev \
        libfftw3-mpi-dev \
        libhwloc-dev \
        libnuma-dev \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        sudo \
        tar \
        unzip \
        wget \
        zsh \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash vscode \
    && echo "vscode ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

ENV CC=mpicc \
    CXX=mpicxx \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none
