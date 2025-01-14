#!/bin/bash
#
#SBATCH --job-name=stabilization
#SBATCH --time=10:00:00
#SBATCH --partition=cpu
#SBATCH --mem=150G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=stabilization.out

cd /hpc/projects/capillary-flow/scripts
module load anaconda
conda activate capillary-flow
echo "Process stabilization data"
srun python process_shaking.py
