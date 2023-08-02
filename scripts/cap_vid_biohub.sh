#!/bin/bash
#
#SBATCH --job-name=cap_vidArray
#SBATCH --time=10:00:00
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --mem=150G
#SBATCH --output=cap_vid_array_%A-%a.out
#SBATCH --array=11-20

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "make cropped capillary videos: participant $SLURM_ARRAY_TASK_ID"
srun python cap_vid_pipeline.py ${SLURM_ARRAY_TASK_ID}
echo "completed cropped capillary videos part$SLURM_ARRAY_TASK_ID"
exit