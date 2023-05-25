#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

srun hostname
srun sleep 60

echo "Current working directory: $PWD"
ls -l
cd /home/marcus.forst
