
#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SRC_DIR}/build}"
INSTALL_DIR="${INSTALL_DIR:-${SRC_DIR}/install}"
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"

echo "==> Source dir : ${SRC_DIR}"
echo "==> Build dir  : ${BUILD_DIR}"
echo "==> Install dir: ${INSTALL_DIR}"
echo "==> Build type : ${BUILD_TYPE}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DIPPL_ENABLE_UNIT_TESTS=ON \
    -DIPPL_ENABLE_TESTS=ON \
    -DIPPL_ENABLE_FFT=ON \
    -DIPPL_ENABLE_ALPINE=ON \
    -DIPPL_ENABLE_SOLVERS=ON \

cmake --build "${BUILD_DIR}" --parallel 4
 
if [[ -f "${BUILD_DIR}/compile_commands.json" ]]; then
    ln -sf "${BUILD_DIR}/compile_commands.json" "${SRC_DIR}/compile_commands.json"
    echo "==> compile_commands.json available at:"
    echo "    ${BUILD_DIR}/compile_commands.json"
    echo "    ${SRC_DIR}/compile_commands.json"
else
    echo "Warning: compile_commands.json was not generated."
fi

echo "==> Build finished successfully."

echo
echo "Optional:"
echo "    ctest --test-dir \"${BUILD_DIR}\" --output-on-failure"
