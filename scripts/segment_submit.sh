#!/bin/bash
#
#SBATCH --job-name=segment_capillaries
#SBATCH --time=00:05:00
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=segment.out

cd /hpc/projects/capillary-flow/src
module load anaconda
conda activate detectron2
echo "segment capillaries"
srun --pty --gpus=1 --partition=gpu python segment.py
echo "completed segmentation"
