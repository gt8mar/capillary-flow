#!/bin/bash
#
#SBATCH --job-name=centerlines
#SBATCH --time=00:10:00
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=centerlines_%A.out
#SBATCH --error=centerlines_%A.err

cd /hpc/projects/capillary-flow/src
module load anaconda
conda activate capillary-flow
echo "find centerlines"
srun python find_centerline.py
echo "completed centerlines"
