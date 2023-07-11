#!/bin/bash
#
#SBATCH --job-name=kymographArray
#SBATCH --time=04:00:00
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --mem=150G
#SBATCH --output=kymo_array_%A-%a.out
#SBATCH --array=11-12

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make centerlines and kymographs: participant $SLURM_ARRAY_TASK_ID"
srun python kymo_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed kymographs part$SLURM_ARRAY_TASK_ID"
exit