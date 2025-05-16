#!/bin/bash -l
#SBATCH --job-name=porkvision
#SBATCH --cluster=gpsc8
#SBATCH --partition=standard
#SBATCH --account=aafc_aac
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/porkvision_%j.log

conda activate porkvision-1.0.0
python ./src/main.py
