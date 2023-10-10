#!/bin/bash
#
#SBATCH --job-name=bkgdArray
#SBATCH --array=21-27
#SBATCH --time=04:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --mem=50G
#SBATCH --output=bkgd_array_%A-%a.out

participants=(23 24 25)
cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make backgrounds: participant ${SLURM_ARRAY_TASK_ID}"
srun python bkgd_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed background part${SLURM_ARRAY_TASK_ID}"
