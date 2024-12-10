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
SCRIPT_DIR="/hpc/projects/capillary-flow/scripts"
PYTHON_SCRIPT="${SCRIPT_DIR}/process_data.py"

srun python "$PYTHON_SCRIPT"
