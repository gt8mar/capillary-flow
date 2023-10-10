#!/bin/bash
#
#SBATCH --job-name=centerlineArray
#SBATCH --time=10:00:00
#SBATCH --partition=cpu
#SBATCH --mem=150G
#SBATCH --output=cent_array_%A-%a.out
#SBATCH --array=9-20

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make centerlines: participant $SLURM_ARRAY_TASK_ID"
srun python centerline_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed centerlines part$SLURM_ARRAY_TASK_ID"
exit