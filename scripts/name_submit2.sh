#!/bin/bash
#
#SBATCH --job-name=capName
#SBATCH --time=10:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=cap_name_array_%A-%a.out
#SBATCH --array=38-38

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "size: participant $SLURM_ARRAY_TASK_ID"
python cap_name_pipeline2.py ${SLURM_ARRAY_TASK_ID}
echo "completed size part$SLURM_ARRAY_TASK_ID"
exit