FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG KOKKOS_REF=4.3.01

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV CONDA_DIR=/opt/conda \
    KOKKOS_SOURCE_DIR=/opt/kokkos-src \
    FETCHCONTENT_SOURCE_DIR_KOKKOS=/opt/kokkos-src \
    PGAS_KOKKOS_ARCH=TURING75 \
    PGAS_CUDA_ARCHITECTURES=75 \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    CXX=/opt/kokkos-src/bin/nvcc_wrapper \
    NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++ \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV PATH=${CONDA_DIR}/bin:${CUDA_HOME}/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        bison \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        file \
        flex \
        g++ \
        gcc \
        gfortran \
        git \
        gperf \
        libdbus-1-3 \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libarmadillo-dev \
        libboost-dev \
        libice6 \
        libgsl-dev \
        libjsoncpp-dev \
        libopenblas-dev \
        libsm6 \
        libtool \
        libx11-6 \
        libxext6 \
        libxkbcommon-x11-0 \
        libxrender1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        ninja-build \
        patchelf \
        perl \
        pkg-config \
        tar \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/c_spikes_requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/c_spikes_requirements.txt \
    && rm /tmp/c_spikes_requirements.txt

RUN rm -rf "${KOKKOS_SOURCE_DIR}" \
    && git clone --depth 1 --branch "${KOKKOS_REF}" https://github.com/kokkos/kokkos.git "${KOKKOS_SOURCE_DIR}"

COPY postInstall /postInstall
RUN bash /postInstall
