#!/bin/bash
#
#SBATCH --job-name=detectron2train
#
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array 0-11
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:1
#
# srun hostname

participants=(9 10 11 12 13 14 15 16 17 18 19 20)
cd /hpc/projects/capillary-flow/scripts
echo "make backgrounds"
python pipeline.py ${participants[$SLURM_ARRAY_TASK_ID]}
echo "completed backgrounds"