#!/bin/bash -l
#SBATCH --job-name=porkvision
#SBATCH --cluster=<cluster name>
#SBATCH --partition=standard
#SBATCH --account=<account name>
#SBATCH --time=<time limit, e.g., 24:00:00 for 24 hours>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<number of CPUs> 
#SBATCH --mem=<amount of memory, e.g., 16G for 16 GB>   
#SBATCH --output=logs/porkvision_%j.log

conda activate porkvision-1.0.0
python ./src/main.py
