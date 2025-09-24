#!/bin/bash -l
#SBATCH --job-name=ifum_0
#SBATCH --output=job-info.out
#SBATCH --error=job-info.err
#SBATCH --account=pi-chard
#SBATCH --partition=caslake
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --output=./jobs/ifum_job_%j.out
#SBATCH --error=./jobs/ifum_job_%j.err

echo "$(date -u '+%Y-%m-%d %H:%M:%S') UTC | job submitted"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on $SLURM_NNODES nodes ($SLURM_JOB_PARTITION)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/babnigg/conda_envs/ifum_parsl
python run.py --nodes $SLURM_NNODES