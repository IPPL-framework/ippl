#!/usr/bin/env bash
set -Eeuo pipefail

# Require CTEST_JOBS to be set explicitly
if [[ -z "${CTEST_JOBS:-}" ]]; then
  echo "❌ ERROR: CTEST_JOBS must be set explicitly (number of processors)."
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  echo "❌ ERROR: 'module' command not found in this environment."
  echo "   Make sure environment modules are available before running."
  exit 1
fi

module purge
module load Stages/2025 GCC OpenMPI CMake NCCL

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export UCC_TL_NCCL_TUNE=allreduce:cuda:inf

mkdir -p "${CTEST_RESULTS_DIR}"

CTEST_ARGS=(
  --test-dir "${BUILD_DIR}"
  --output-on-failure
  -j "${CTEST_JOBS}"
  --no-tests=error
  --output-junit "${CTEST_RESULTS_DIR}/ctest-report.xml"
)

if [[ -n "${CTEST_LABELS:-}" ]]; then
  CTEST_ARGS=(-L "${CTEST_LABELS}" "${CTEST_ARGS[@]}")
fi

echo "Running ctest with args: ${CTEST_ARGS[*]}"

set +e
ctest "${CTEST_ARGS[@]}" 2>&1 | tee "${CTEST_RESULTS_DIR}/ctest.stdout.log"
CTEST_STATUS=${PIPESTATUS[0]}
set -e

exit "${CTEST_STATUS}"
