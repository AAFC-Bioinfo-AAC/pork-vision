#!/bin/bash -l
#SBATCH --job-name=porkvision
#SBATCH --output=porkvision.out
#SBATCH --cluster=gpsc8
#SBATCH --partition=standard
#SBATCH --account=aafc_aac
#SBATCH --time=02:30:00
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --comment="registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04,tmpfs_size=2G"
#SBATCH --mail-type=END,FAIL,BEGIN

conda activate porkvision-1.0.0
python ./src/main.py
