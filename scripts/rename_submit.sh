#!/bin/bash
#
#SBATCH --job-name=capRename
#SBATCH --time=10:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=cap_rename_array_%A-%a.out
#SBATCH --array=28-32

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "size: participant $SLURM_ARRAY_TASK_ID"
python cap_rename_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed size part$SLURM_ARRAY_TASK_ID"
exit