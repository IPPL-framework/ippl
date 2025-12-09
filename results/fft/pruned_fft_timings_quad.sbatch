#!/bin/bash
#SBATCH --job-name=scaling_test
#SBATCH --account=c41
#SBATCH --time=00:20:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=scaling_test.out
#SBATCH --error=scaling_test.error

export BASE_DIR=base_dir
export JOB_DIR=job_dir
export WRAPPER=../mpi_wrapper.sh

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

# Create output directory with timestamp
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

for num_gpus in 64 128 256; do
    num_nodes=$(( (num_gpus + 3) / 4 ))
    if [ $num_nodes -lt 1 ]; then
        num_nodes=1
    fi
    
    if [ $num_gpus -lt 4 ]; then
        tasks_per_node=$num_gpus
    else
        tasks_per_node=4
    fi
    
    total_tasks=$num_gpus
    
    echo "=============================================="
    echo "Running with $num_gpus GPUs ($num_nodes nodes, $tasks_per_node tasks/node)"
    echo "=============================================="
    
    for num_concurrent in 1 2 4 8; do
        CSV_FILE="${OUTPUT_DIR}/timings_gpus${num_gpus}_conc${num_concurrent}.csv"
        
        echo "Testing --concurrent $num_concurrent -> $CSV_FILE"
        
        srun --nodes=$num_nodes \
             --ntasks=$total_tasks \
             --ntasks-per-node=$tasks_per_node \
             --gpus-per-task=1 \
             $WRAPPER ./bin/TestPrunedFFTPerformance \
             --concurrent $num_concurrent \
             --csv "$CSV_FILE"
    done
done

echo "=============================================="
echo "Scaling test complete!"
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"
echo "=============================================="
