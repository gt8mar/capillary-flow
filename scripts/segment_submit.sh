#!/bin/bash
#
#SBATCH --job-name=segment_capillaries
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --mem=2G
#SBATCH --output=segment.out

cd /hpc/projects/capillary-flow/src
echo "segment capillaries"
srun python segment.py
echo "completed segmentation"
