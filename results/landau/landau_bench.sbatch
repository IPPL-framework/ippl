#!/bin/bash
#SBATCH --job-name=landau_damping_test
#SBATCH --account=c41
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=landau_damping_test.out
#SBATCH --error=landau_damping_test.error

export BASE_DIR=base_dir
export JOB_DIR=job_dir
export WRAPPER=../mpi_wrapper.sh

mkdir -p data

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

CSV_FILE="landau_damping_results.csv"

# Grid size
GRID_X=256
GRID_Y=256
GRID_Z=256

# 10 particles per grid cell
NUM_PARTICLES=$((GRID_X * GRID_Y * GRID_Z * 10))

# Other parameters
NT=20
DT=0.05
SHAPE="B-spline"
SHAPE_ORDER=1
TOLERANCE="1e-4"

# Write CSV header
echo "variant,num_gpus,num_nodes,mainTimer_s,particlesCreation_s,dumpData_s,kick_s,drift_s,particleBC_s,initializeShapeFunc_s,NativeNUFFT_init_s,updateParticle_s,ScatterPIFNUFFT_s,scatterKernel_s,GatherPIFNUFFT_s,fillHalo_s,gatherKernel_s" > "$CSV_FILE"

# Function to parse timing output and append to CSV
parse_and_append() {
    local VARIANT=$1
    local NUM_GPUS=$2
    local NUM_NODES=$3
    local OUTPUT="$4"
    
    # Extract timing for each component (using Wall max for parallel runs)
    mainTimer=$(echo "$OUTPUT" | grep "mainTimer" | awk '{print $4}')
    particlesCreation=$(echo "$OUTPUT" | grep "particlesCreation" | awk '{print $5}')
    dumpData=$(echo "$OUTPUT" | grep "dumpData" | awk '{print $5}')
    kick=$(echo "$OUTPUT" | grep "kick\.\.\.\." | awk '{print $5}')
    drift=$(echo "$OUTPUT" | grep "drift\.\.\.\." | awk '{print $5}')
    particleBC=$(echo "$OUTPUT" | grep "particleBC" | awk '{print $5}')
    initializeShapeFunc=$(echo "$OUTPUT" | grep "initializeShapeFunc" | awk '{print $5}')
    NativeNUFFT_init=$(echo "$OUTPUT" | grep "NativeNUFFT::init" | awk '{print $5}')
    updateParticle=$(echo "$OUTPUT" | grep "updateParticle" | awk '{print $5}')
    ScatterPIFNUFFT=$(echo "$OUTPUT" | grep "ScatterPIFNUFFT" | awk '{print $5}')
    scatterKernel=$(echo "$OUTPUT" | grep "scatterKernel" | awk '{print $5}')
    GatherPIFNUFFT=$(echo "$OUTPUT" | grep "GatherPIFNUFFT" | awk '{print $5}')
    fillHalo=$(echo "$OUTPUT" | grep "fillHalo" | awk '{print $5}')
    gatherKernel=$(echo "$OUTPUT" | grep "gatherKernel" | awk '{print $5}')
    
    # Append to CSV
    echo "$VARIANT,$NUM_GPUS,$NUM_NODES,$mainTimer,$particlesCreation,$dumpData,$kick,$drift,$particleBC,$initializeShapeFunc,$NativeNUFFT_init,$updateParticle,$ScatterPIFNUFFT,$scatterKernel,$GatherPIFNUFFT,$fillHalo,$gatherKernel" >> "$CSV_FILE"
}

# Loop through GPU counts: 1, 2, 4, 8, 16, 32, 64, 128
for num_gpus in 1 2 4 8; do
    # Calculate number of nodes (minimum 1, 4 GPUs per node)
    num_nodes=$(( (num_gpus + 3) / 4 ))
    if [ $num_nodes -lt 1 ]; then
        num_nodes=1
    fi
    
    # Calculate tasks per node
    if [ $num_gpus -lt 4 ]; then
        tasks_per_node=$num_gpus
    else
        tasks_per_node=4
    fi
    
    total_tasks=$num_gpus
    
    echo "=============================================="
    echo "Running with $num_gpus GPUs ($num_nodes nodes, $tasks_per_node tasks/node)"
    echo "Grid: ${GRID_X}x${GRID_Y}x${GRID_Z}, Particles: $NUM_PARTICLES"
    echo "=============================================="
    
    # Run the standard (non-pruned) benchmark
    echo "--- Running LandauDampingPIF (standard) ---"
    OUTPUT_STANDARD=$(srun --nodes=$num_nodes \
                  --ntasks=$total_tasks \
                  --ntasks-per-node=$tasks_per_node \
                  --gpus-per-task=1 \
                  $WRAPPER ./bin/LandauDampingPIF $GRID_X $GRID_Y $GRID_Z $NUM_PARTICLES $NT $DT $SHAPE $SHAPE_ORDER $TOLERANCE --info 5 2>&1)
    
    echo "$OUTPUT_STANDARD"
    parse_and_append "standard" "$num_gpus" "$num_nodes" "$OUTPUT_STANDARD"
    echo "Standard results appended to $CSV_FILE"
    echo ""
    
    # Run the pruned benchmark
    echo "--- Running LandauDampingPIFPruned ---"
    OUTPUT_PRUNED=$(srun --nodes=$num_nodes \
                  --ntasks=$total_tasks \
                  --ntasks-per-node=$tasks_per_node \
                  --gpus-per-task=1 \
                  $WRAPPER ./bin/LandauDampingPIFPruned $GRID_X $GRID_Y $GRID_Z $NUM_PARTICLES $NT $DT $SHAPE $SHAPE_ORDER $TOLERANCE --info 5 2>&1)
    
    echo "$OUTPUT_PRUNED"
    parse_and_append "pruned" "$num_gpus" "$num_nodes" "$OUTPUT_PRUNED"
    echo "Pruned results appended to $CSV_FILE"
    echo ""
done

echo "=============================================="
echo "Landau Damping benchmark complete!"
echo "Results saved to: $CSV_FILE"
echo "=============================================="
cat "$CSV_FILE"
