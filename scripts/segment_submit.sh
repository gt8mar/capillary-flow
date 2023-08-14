#!/bin/bash
#
#SBATCH --job-name=segment_capillaries
#SBATCH --time=00:10:00
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=segment_%A.out
#SBATCH --error=segment_%A.err

cd /hpc/projects/capillary-flow/src
module load anaconda cuda cudnn gcc
conda activate detectron3
echo "segment capillaries"
srun python segment.py
echo "completed segmentation"
