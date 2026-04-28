FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MINIFORGE_VERSION=24.11.3-0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER 0:0

ENV CONDA_DIR=/opt/conda \
    VCPKG_ROOT=/opt/vcpkg \
    VCPKG_DEFAULT_TRIPLET=x64-linux \
    KOKKOS_SOURCE_DIR=/opt/kokkos-src \
    FETCHCONTENT_SOURCE_DIR_KOKKOS=/opt/kokkos-src \
    PGAS_KOKKOS_ARCH=TURING75 \
    PGAS_CUDA_ARCHITECTURES=75 \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    CXX=/opt/kokkos-src/bin/nvcc_wrapper \
    NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++ \
    CMAKE_PREFIX_PATH=/opt/vcpkg/installed/x64-linux \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VCPKG_DISABLE_METRICS=1

ENV NVIDIA_PYTHON_LIBS=/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cusolver/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nvtx/lib
ENV LD_LIBRARY_PATH=/opt/vcpkg/installed/x64-linux/lib:${NVIDIA_PYTHON_LIBS}:/usr/local/cuda/lib64
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
        libice6 \
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

RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && mamba install -y python=3.11 pip \
    && mamba clean -afy

COPY requirements.txt /tmp/c_spikes_requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/c_spikes_requirements.txt \
    && rm /tmp/c_spikes_requirements.txt

COPY postInstall /postInstall
RUN chmod +x /postInstall \
    && /postInstall
