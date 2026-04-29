#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

export OMPI_ALLOW_RUN_AS_ROOT="${OMPI_ALLOW_RUN_AS_ROOT:-1}"
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="${OMPI_ALLOW_RUN_AS_ROOT_CONFIRM:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export KOKKOS_NUM_THREADS="${KOKKOS_NUM_THREADS:-2}"

cd "${repo_root}"

if [[ ! -f build/devcontainer/CMakeCache.txt ]]; then
    .devcontainer/scripts/configure.sh
fi

cmake --build build/devcontainer --target FFT --parallel "${IPPL_BUILD_JOBS:-2}"
ctest --preset devcontainer-smoke "$@"
