#!/bin/bash
#
#SBATCH --job-name=analysisArray
#SBATCH --array=23-25
#SBATCH --time=04:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --output=array_%A-%a.out

participants=(23 24 25)
cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make backgrounds: participant ${SLURM_ARRAY_TASK_ID}"
srun python pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed background part${SLURM_ARRAY_TASK_ID}"
