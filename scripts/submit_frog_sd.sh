#!/bin/bash
#
#SBATCH --job-name=frog_sd
#SBATCH --time=5-00:00:00
#SBATCH --mem=250G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output_frog_sd_%A.out

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "making frog standard deviations"
python frog_sd_pipeline.py
echo "completed frog standard deviations"
exit