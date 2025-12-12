#!/bin/bash
#
# run_breakdown_sweep.sh
# 
# Runs NUFFT breakdown benchmark across multiple tolerances (widths).
# Outputs CSV files for each configuration.
#
# Usage:
#   ./run_breakdown_sweep.sh [grid_size] [rho] [warmup] [runs]
#   ./run_breakdown_sweep.sh 256 10 5 20

GRID=${1:-256}
RHO=${2:-10}
WARMUP=${3:-5}
RUNS=${4:-20}

# Tolerances corresponding to different kernel widths
# tol=1e-2 -> w=2, tol=1e-3 -> w=3, ..., tol=1e-7 -> w=7
TOLERANCES="1e-2 1e-3 1e-4 1e-5 1e-6 1e-7"

OUTPUT_DIR="breakdown_results"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "NUFFT Breakdown Sweep"
echo "========================================"
echo "Grid:      ${GRID}^3"
echo "Rho:       ${RHO}"
echo "Warmup:    ${WARMUP}"
echo "Runs:      ${RUNS}"
echo "Tolerances: ${TOLERANCES}"
echo "Output:    ${OUTPUT_DIR}/"
echo "========================================"
echo ""

for TOL in $TOLERANCES; do
    echo "----------------------------------------"
    echo "Running tolerance = ${TOL}"
    echo "----------------------------------------"
    
    # Run benchmark
    ./BenchmarkNUFFTBreakdown ${GRID} ${RHO} ${TOL} ${WARMUP} ${RUNS}
    
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

for TOL in $TOLERANCES; do
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
