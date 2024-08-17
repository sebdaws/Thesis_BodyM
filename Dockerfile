ARG IMAGE_NAME
FROM ${IMAGE_NAME}:12.5.1-base-ubuntu24.04 as base

ENV NV_CUDA_LIB_VERSION 12.5.1-1

FROM base as base-amd64

ENV NV_NVTX_VERSION 12.5.82-1
ENV NV_LIBNPP_VERSION 12.3.0.159-1
ENV NV_LIBNPP_PACKAGE libnpp-12-5=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.5.1.3-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-5
ENV NV_LIBCUBLAS_VERSION 12.5.3.2-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

FROM base as base-arm64

ENV NV_NVTX_VERSION 12.5.82-1
ENV NV_LIBNPP_VERSION 12.3.0.159-1
ENV NV_LIBNPP_PACKAGE libnpp-12-5=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.5.1.3-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-5
ENV NV_LIBCUBLAS_VERSION 12.5.3.2-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-5=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-12-5=${NV_NVTX_VERSION} \
    libcusparse-12-5=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME}

# Add entrypoint items
COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]