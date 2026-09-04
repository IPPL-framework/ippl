#!/usr/bin/env bash
# Build the bp2vtk converter against the ippl in-tree ADIOS2.
#
#   ./build.sh [<ippl-build-dir>]
#
# Then convert a run:
#   ./_b/bp2vtk /path/to/alpine.bp [output_dir]
# and open the generated *.pvd files in ParaView.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPPL_BUILD="${1:-${SCRIPT_DIR}/../../build}"

# Locate the ADIOS2 CMake package produced by the ippl build.
ADIOS2_DIR=""
for cand in \
    "${IPPL_BUILD}/_deps/adios2-build" \
    "${IPPL_BUILD}/ippl-cmake-shims/adios2"; do
  if [[ -f "${cand}/adios2-config.cmake" ]]; then
    ADIOS2_DIR="${cand}"
    break
  fi
done
if [[ -z "${ADIOS2_DIR}" ]]; then
  echo "build.sh: could not find adios2-config.cmake under ${IPPL_BUILD}." >&2
  echo "          Pass the ippl build dir explicitly: ./build.sh /path/to/ipplADIOS/build" >&2
  exit 1
fi

# Prefer the MPICH wrappers used elsewhere in this project, if present.
MPICC="${MPICC:-$(command -v mpicc || echo /usr/lib64/mpich/bin/mpicc)}"
MPICXX="${MPICXX:-$(command -v mpicxx || echo /usr/lib64/mpich/bin/mpicxx)}"

echo "bp2vtk: ADIOS2_DIR=${ADIOS2_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/_b" \
  -DADIOS2_DIR="${ADIOS2_DIR}" \
  -DCMAKE_C_COMPILER="${MPICC}" \
  -DCMAKE_CXX_COMPILER="${MPICXX}"
cmake --build "${SCRIPT_DIR}/_b" -j

echo
echo "Built ${SCRIPT_DIR}/_b/bp2vtk"
echo "Usage: ${SCRIPT_DIR}/_b/bp2vtk <input.bp> [output_dir]"
