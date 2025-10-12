#!/bin/bash -e
#PBS -A workflow_scaling
#PBS -N ifum_agents
#PBS -q prod
#PBS -l select=2
#PBS -l walltime=04:00:00
#PBS -l filesystems=flare
#PBS -l place=scatter
#PBS -j oe

create_run_dir () {
    # Define the base name for directories
    base_name="runs/run-"

    # Find the highest numbered directory
    last_dir=$(ls -d ${base_name}[0-9]* 2>/dev/null | sort -V | tail -n 1)

    # Extract the number and increment it
    if [[ -z "$last_dir" ]]; then
    next_num=1
    else
    numbers=$(echo "$last_dir" | grep -o '[0-9]\+')
    next_num=$(( $((10#$numbers)) + 1 ))
    fi

    # Format the number with leading zeros (e.g., 001, 002)
    next_dir=$(printf "%s%03d" "$base_name" "$next_num")

    # Create the new directory
    mkdir -p "$next_dir"
    echo $next_dir
}


module load frameworks
conda activate /flare/workflow_scaling/alokvk2/agents/ipdps26-ifum-academy/conda_env
source /flare/workflow_scaling/alokvk2/agents/ipdps26-ifum-academy/venv/bin/activate
cd /flare/workflow_scaling/alokvk2/agents/ipdps26-ifum-academy

HEAD_NODE_IP=$(getent hosts "$(hostname).hsn.cm.aurora.alcf.anl.gov" | awk '{ print $1 }' | sort | head -n 1)
REDIS_PORT=6389
redis-server --port $REDIS_PORT --save "" --appendonly no --protected-mode no &> /dev/null &
REDIS=$!
echo "Redis server started"

run_dir=$(create_run_dir)

python -m ifum_agent.run \
       --input-dir /flare/workflow_scaling/alokvk2/agents/ipdps26-ifum-academy/data \
       --run-dir $run_dir \
       --parsl htex-aurora-cpu \
       --workers-per-node 51 \
       --redis-host $HEAD_NODE_IP \
       --redis-port $REDIS_PORT

# Shutdown services
kill $REDIS
