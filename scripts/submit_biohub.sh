#!/bin/bash
#
#SBATCH --job-name=detectron2train
#
#SBATCH --time=00:10:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#
# srun hostname
cd /home/marcus.forst
echo "begin training model"
python segment.py
echo "completed training model"