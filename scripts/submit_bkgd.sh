#!/bin/bash
#
#SBATCH --job-name=bkgdArray
#SBATCH --array=26-27
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --output=bkgd_array_%A-%a.out

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make backgrounds: participant ${SLURM_ARRAY_TASK_ID}"
srun python bkgd_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed background part${SLURM_ARRAY_TASK_ID}"
