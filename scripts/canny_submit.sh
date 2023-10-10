#!/bin/bash
#
#SBATCH --job-name=cannyArray
#SBATCH --time=10:00:00
#SBATCH --partition=cpu
#SBATCH --mem=150G
#SBATCH --output=canny_array_%A-%a.out
#SBATCH --array=9-20

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make velocities: participant $SLURM_ARRAY_TASK_ID"
srun python canny_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed velocities part$SLURM_ARRAY_TASK_ID"
exit