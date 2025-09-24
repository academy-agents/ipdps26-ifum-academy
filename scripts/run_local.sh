create_run_dir () {
    # Define the base name for directories
    base_name="dir"

    # Find the highest numbered directory
    last_dir=$(ls -d ${base_name}[0-9]* 2>/dev/null | sort -V | tail -n 1)

    # Extract the number and increment it
    if [[ -z "$last_dir" ]]; then
    next_num=1
    else
    next_num=$(( ${last_dir##${base_name}} + 1 ))
    fi

    # Format the number with leading zeros (e.g., 001, 002)
    next_dir=$(printf "%s%03d" "$base_name" "$next_num")

    # Create the new directory
    mkdir "$next_dir"
    return $next_dir
}

redis-server --bind 0.0.0.0 --appendonly no --logfile redis.log --protected-mode no &
redis_pid=$!
echo launched redis on $redis_pid

create_run_dir
run_dir=$?

python -m ifum_agents.run \
       --input-dir /home/alokvk2/research/agents/IFU-M/data \
       --run_dir $run_dir \
       --parsl htex-local \
       --workers-per-node 8 \
       --redis-host 127.0.0.1 \
       --redis-port 6379

echo Python done

# Shutdown services
kill $redis_pid