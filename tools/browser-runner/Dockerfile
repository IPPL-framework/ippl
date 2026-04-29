FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        gcc \
        git \
        libhwloc-dev \
        libnuma-dev \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ippl-src
COPY . .

RUN cmake -S . -B /opt/ippl-build -G Ninja \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX=/opt/ippl \
        -DIPPL_ENABLE_ALPINE=OFF \
        -DIPPL_ENABLE_COSMOLOGY=OFF \
        -DIPPL_ENABLE_EXAMPLES=OFF \
        -DIPPL_ENABLE_FFT=OFF \
        -DIPPL_ENABLE_SOLVERS=OFF \
        -DIPPL_ENABLE_TESTS=OFF \
        -DIPPL_ENABLE_UNIT_TESTS=OFF \
        -DIPPL_PLATFORMS=SERIAL \
        -DIPPL_USE_STANDARD_FOLDERS=ON \
        -DKokkos_VERSION=git.4.7.02 \
    && cmake --build /opt/ippl-build --target install --parallel 2

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        gcc \
        libhwloc15 \
        libnuma1 \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/ippl /opt/ippl
COPY tools/browser-runner/runner.py /opt/ippl-runner/runner.py

ENV IPPL_PREFIX=/opt/ippl \
    LD_LIBRARY_PATH=/opt/ippl/lib \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none

EXPOSE 5050

CMD ["python3", "/opt/ippl-runner/runner.py", "--host", "0.0.0.0", "--port", "5050"]
