#!/bin/bash
# -----------------------------------------------------------------------------
# Run LandauDampingBench and optionally upload results to bencher.dev.
#
# Environment variables expected:
#   BUILD_PATH          - path to the CMake build directory
#   SRUN_FLAGS          - flags for srun (may be empty)
#   BENCHMARK_MIN_TIME  - minimum benchmark run time (default: 1.0s)
#   BENCHER_PROJECT     - bencher.dev project slug
#   BENCHER_TESTBED     - bencher.dev testbed name
#   BENCHER_API_KEY     - optional bencher.dev API key (preferred)
#   BENCHER_API_TOKEN   - optional bencher.dev API token (fallback)
#   CI_COMMIT_REF_NAME  - Git branch/tag name
# -----------------------------------------------------------------------------

set -euo pipefail

BENCHMARK_MIN_TIME="${BENCHMARK_MIN_TIME:-1.0s}"

# Locate the benchmark binary anywhere under the build tree.
BENCH_BIN=$(find "$BUILD_PATH" -name LandauDampingBench -type f -executable | head -n 1)

if [ -z "$BENCH_BIN" ]; then
  echo "LandauDampingBench binary not found in $BUILD_PATH"
  echo "Contents of likely locations:"
  ls -la "$BUILD_PATH/bin" 2>/dev/null || true
  ls -la "$BUILD_PATH/demos/alpine" 2>/dev/null || true
  echo "All executables matching *Bench*:"
  find "$BUILD_PATH" -type f -executable -name "*Bench*" 2>/dev/null | head -20
  exit 1
fi

echo "Using benchmark binary: $BENCH_BIN"

# shellcheck disable=SC2086
# --cpu-bind=none avoids conflicts with the outer SLURM allocation's CPU binding.
srun --cpu-bind=none $SRUN_FLAGS \
  "$BENCH_BIN" \
  16 16 16 100000 5 FFT 0.01 LeapFrog \
  --overallocate 2.0 --info 10 \
  --benchmark_out=LandauDampingBench.json \
  --benchmark_min_time="$BENCHMARK_MIN_TIME"

ls -la LandauDampingBench.json LandauDampingBench.bmf.json || true

# Determine bencher credentials. Prefer BENCHER_API_KEY; fall back to BENCHER_API_TOKEN.
# The flag is chosen by variable name: API keys use --key, tokens use --token.
BENCHER_CRED=""
BENCHER_CRED_FLAG=""
if [ -n "${BENCHER_API_KEY:-}" ]; then
  BENCHER_CRED="$BENCHER_API_KEY"
  BENCHER_CRED_FLAG="--key"
  unset BENCHER_API_TOKEN
elif [ -n "${BENCHER_API_TOKEN:-}" ]; then
  BENCHER_CRED="$BENCHER_API_TOKEN"
  BENCHER_CRED_FLAG="--token"
fi

if [ -z "$BENCHER_CRED" ]; then
  echo "Neither BENCHER_API_KEY nor BENCHER_API_TOKEN set; skipping bencher upload"
fi

# Only upload from rank 0 to avoid duplicate submissions in multi-rank runs.
if [ "${SLURM_PROCID:-0}" != "0" ] && [ "${PMI_RANK:-0}" != "0" ] && [ "${OMPI_COMM_WORLD_RANK:-0}" != "0" ]; then
  echo "Skipping bencher upload on rank ${SLURM_PROCID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
  BENCHER_CRED_FLAG=""
fi

if [ -n "$BENCHER_CRED_FLAG" ]; then
  echo "Uploading to bencher.dev using $BENCHER_CRED_FLAG"

  bencher run \
    --project "$BENCHER_PROJECT" \
    --testbed "$BENCHER_TESTBED" \
    --adapter cpp_google \
    --file LandauDampingBench.json \
    --branch "$CI_COMMIT_REF_NAME" \
    "$BENCHER_CRED_FLAG" "$BENCHER_CRED"

  bencher run \
    --project "$BENCHER_PROJECT" \
    --testbed "$BENCHER_TESTBED" \
    --adapter json \
    --file LandauDampingBench.bmf.json \
    --branch "$CI_COMMIT_REF_NAME" \
    "$BENCHER_CRED_FLAG" "$BENCHER_CRED"
fi
