#!/bin/bash
#
#SBATCH --job-name=vel_ml
#SBATCH --time=5-00:00:00
#SBATCH --mem=250G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output_ml_%A.out

cd /hpc/projects/capillary-flow/src
module load anaconda
conda activate capillary-flow
echo "mling"
accelerate launch train.py
echo "completed mling"
exit