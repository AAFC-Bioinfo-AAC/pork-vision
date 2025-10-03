#!/bin/bash -l

# ---------- SBATCH directives ----------

#SBATCH --job-name=porkvision
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=<number of CPUs> 
#SBATCH --mem=<amount of memory>            # e.g., 16G for 16 GB   
#SBATCH --time=<time limit>                 # e.g., 24:00:00 for 24 hours

#SBATCH --cluster=<cluster name>
#SBATCH --partition=<partition name>
#SBATCH --account=<account name>

# Put Slurm logs in the submit directory
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


# ---------- Project-specific env ----------

# Add path to ImageJ executable
export FIJI_CMD="/your/path/to/porkvision-conda-env/bin/ImageJ"

# ---------- Conda activation ----------

# source ~/miniconda3/etc/profile.d/conda.sh
conda activate porkvision-1.0.0

# ---------- Run ----------

# Use srun so Slurm tracks resources and signals properly
echo "[$(date)] Launching main program…"
srun python ./src/main.py

echo "[$(date)] Job complete."
