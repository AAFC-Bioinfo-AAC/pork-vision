#!/bin/bash -l
#SBATCH --job-name=porkvision
#SBATCH --cluster=<cluster name>
#SBATCH --partition=<partition name>
#SBATCH --account=<account name>
#SBATCH --time=<time limit, e.g., 24:00:00 for 24 hours>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<number of CPUs> 
#SBATCH --mem=<amount of memory, e.g., 16G for 16 GB>   
#SBATCH --output=logs/porkvision_%j.log
#SBATCH --error=logs/porkvision_%j.err

# Add path to ImageJ executable
export FIJI_CMD="/your/path/to/porkvision-env/bin/ImageJ"

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate porkvision-1.0.0

# Run the script
python ./src/main.py
