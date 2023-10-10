#!/bin/bash
#
#SBATCH --job-name=capSize
#SBATCH --time=10:00:00
#SBATCH --mem=150G
#SBATCH --output=cap_size_array_%A-%a.out
#SBATCH --array=9-20

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "size: participant $SLURM_ARRAY_TASK_ID"
srun python pipeline_size.py ${SLURM_ARRAY_TASK_ID}
echo "completed size part$SLURM_ARRAY_TASK_ID"
exit