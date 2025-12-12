#!/bin/bash
#
# run_breakdown_sweep.sh
# 
# Runs NUFFT breakdown benchmark across multiple grid sizes (N).
# Outputs CSV files for each configuration.
#
# Usage:
#   ./run_breakdown_sweep.sh [rho] [tolerance] [warmup] [runs]
#   ./run_breakdown_sweep.sh 10 1e-4 5 20

RHO=${1:-10}
TOL=${2:-1e-4}
WARMUP=${3:-5}
RUNS=${4:-20}

# Grid sizes to test
GRID_SIZES="64 128 192 256 320 384"

OUTPUT_DIR="breakdown_results"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "NUFFT Breakdown Sweep (vs Grid Size)"
echo "========================================"
echo "Rho:        ${RHO}"
echo "Tolerance:  ${TOL}"
echo "Warmup:     ${WARMUP}"
echo "Runs:       ${RUNS}"
echo "Grid sizes: ${GRID_SIZES}"
echo "Output:     ${OUTPUT_DIR}/"
echo "========================================"
echo ""

for GRID in $GRID_SIZES; do
    echo "----------------------------------------"
    echo "Running grid size = ${GRID}^3"
    echo "----------------------------------------"
    
    # Run benchmark
    ./test/FFT/BreakdownNUFFT ${GRID} ${RHO} ${TOL} ${WARMUP} ${RUNS}
    
    # Move output CSV to results directory with descriptive name
    CSV_FILE="nufft_breakdown_${GRID}_rho${RHO}.csv"
    if [ -f "$CSV_FILE" ]; then
        mv "$CSV_FILE" "${OUTPUT_DIR}/breakdown_grid${GRID}_rho${RHO}_tol${TOL}.csv"
        echo "Saved: ${OUTPUT_DIR}/breakdown_grid${GRID}_rho${RHO}_tol${TOL}.csv"
    fi
    
    echo ""
done

# Combine all CSVs into one file for easier plotting
echo "Combining results..."
COMBINED="${OUTPUT_DIR}/breakdown_combined.csv"

# Write header
echo "grid,rho,tolerance,timer,run,time_s" > "$COMBINED"

for GRID in $GRID_SIZES; do
    CSV="${OUTPUT_DIR}/breakdown_grid${GRID}_rho${RHO}_tol${TOL}.csv"
    if [ -f "$CSV" ]; then
        # Skip header, add grid/rho/tol columns
        tail -n +2 "$CSV" | while IFS=, read -r timer rank run time; do
            echo "${GRID},${RHO},${TOL},${timer},${run},${time}"
        done >> "$COMBINED"
    fi
done

echo "Combined results: ${COMBINED}"
echo ""
echo "Done! Use plot_nufft_breakdown.py to visualize."
