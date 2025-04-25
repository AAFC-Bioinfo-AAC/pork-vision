#!/bin/bash -l
#SBATCH --job-name=porkvision
#SBATCH --output=output/log.out
#SBATCH --cluster=gpsc7
#SBATCH --partition=gpu_a100
#SBATCH --account=aafc_aac__gpu_a100
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --comment="registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04,tmpfs_size=2G"

conda activate porkvision-1.0.0
python ./src/main.py

