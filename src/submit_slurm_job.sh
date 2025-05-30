#!/usr/bin/env bash
# =============================================================================
#  submit_slurm_job.sh — Submission launcher for SLURM jobs using external config
# -----------------------------------------------------------------------------
#  Loads SLURM environment variables from `slurm.env` and submits the batch job
#  script (e.g., main.py) with those variables made available to SBATCH.
#  Also supports pipeline-level configuration via config files.
# -----------------------------------------------------------------------------
#  Usage:
#    ./run_slurm.sh porkvision.job pipeline.conf
# -----------------------------------------------------------------------------

set -euo pipefail

SLURM_ENV_FILE="slurm.env"
SBATCH_SCRIPT="${1:-porkvision.job}"
PIPELINE_CONFIG="${2:-pipeline.conf}"

if [[ ! -f "$SLURM_ENV_FILE" ]]; then
  echo "[ERROR] SLURM environment file '$SLURM_ENV_FILE' not found."
  exit 1
fi
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[ERROR] SBATCH script '$SBATCH_SCRIPT' not found."
  exit 1
fi
if [[ ! -f "$PIPELINE_CONFIG" ]]; then
  echo "[WARN] Pipeline config '$PIPELINE_CONFIG' not found. Continuing without it."
fi

# Load SLURM configuration
set -a
# shellcheck disable=SC1090
source "$SLURM_ENV_FILE"
set +a

# Submit the SLURM job
sbatch \
  --job-name="${CI_JOB_NAME:-porkvision}" \
  --cluster="$SLURM_CLUSTER" \
  --partition="$SLURM_PARTITION" \
  --account="$SLURM_ACCOUNT" \
  --time="$SLURM_TIME" \
  "$SBATCH_SCRIPT" "$PIPELINE_CONFIG"
