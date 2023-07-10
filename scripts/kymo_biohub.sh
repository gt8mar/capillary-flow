#!/bin/bash
#
#SBATCH --job-name=kymographArray
#SBATCH --array=11-25
#SBATCH --time=04:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --output=kymo_array_%A-%a.out

cd /hpc/projects/capillary-flow/scripts
echo "make centerlines and kymographs: participant ${SLURM_ARRAY_TASK_ID}"
srun python pipeline_kymo.py ${SLURM_ARRAY_TASK_ID}
echo "completed kymographs part${SLURM_ARRAY_TASK_ID}"
