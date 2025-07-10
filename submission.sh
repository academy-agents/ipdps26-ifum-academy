#!/bin/bash
#SBATCH --job-name=ifum_0
#SBATCH --output=job-info.out
#SBATCH --error=job-info.err
#SBATCH --account=pi-chard
#SBATCH --partition=caslake
#SBATCH --time=03:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14

touch $SLURM_JOB_ID.txt
echo "Job ID: $SLURM_JOB_ID" >> $SLURM_JOB_ID.txt
echo "Job name: $SLURM_JOB_NAME" >> $SLURM_JOB_ID.txt
echo "N tasks: $SLURM_ARRAY_TASK_COUNT" >> $SLURM_JOB_ID.txt
echo "N cores: $SLURM_CPUS_ON_NODE" >> $SLURM_JOB_ID.txt
echo "N threads per core: $SLURM_THREADS_PER_CORE" >> $SLURM_JOB_ID.txt
echo "Minimum memory required per CPU: $SLURM_MEM_PER_CPU" >> $SLURM_JOB_ID.txt
echo "Requested memory per GPU: $SLURM_MEM_PER_GPU" >> $SLURM_JOB_ID.txt

module load python
conda activate /home/babnigg/conda_envs/parsl_py38
python run.py