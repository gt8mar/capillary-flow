#!/bin/bash
#
#SBATCH --job-name=kymographArray
#SBATCH --time=10:00:00
#SBATCH --mem=150G
#SBATCH --output=kymo_array_%A-%a.out
#SBATCH --array=9-20

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make kymographs: participant $SLURM_ARRAY_TASK_ID"
srun python kymo_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed kymographs part$SLURM_ARRAY_TASK_ID"
exit
