#!/bin/bash -l
#SBATCH --job-name=ifum_0
#SBATCH --output=job-info.out
#SBATCH --error=job-info.err
#SBATCH --account=pi-chard
#SBATCH --partition=caslake
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=21
#SBATCH --output=./jobs/ifum_job_%j.out
#SBATCH --error=./jobs/ifum_job_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/babnigg/conda_envs/ifum_parsl
python run.py