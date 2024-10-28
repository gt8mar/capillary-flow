#!/bin/bash
#
#SBATCH --job-name=capillary_velocity_analysis
#SBATCH --time=01:00:00
#SBATCH --gpus=1    # Changed to 1 GPU per node
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Changed to match number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --output=capillary_%A.out
#SBATCH --error=capillary_%A.err

# Change to the project directory
cd /hpc/projects/capillary-flow/src || exit 1

# Load required modules
module load anaconda cuda cudnn gcc

# Activate the conda environment
source activate capillary-flow

# Create results directory if it doesn't exist
mkdir -p /hpc/projects/capillary-flow/frog/results

# Get the number of available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of available GPUs: $NUM_GPUS"

# Launch one Python process
python metha_velocities_gpu.py 0 1

echo "Capillary velocity analysis completed"