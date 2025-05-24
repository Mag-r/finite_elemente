#!/bin/bash

PROCESS_COUNTS=(1 2 4 8 16)
LOG_FILE="runtime_results.txt"
> "$LOG_FILE"

TOTAL_CORES=$(nproc)
declare -A PIDS
declare -A CORES_USED
declare -A START_TIMES
declare -A NP_FOR_PID

# Cleanup background jobs on exit
cleanup() {
    for pid in "${!PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
}
trap cleanup EXIT

# Waits for enough cores to be free
wait_for_cores() {
    local needed=$1
    while true; do
        local used=0
        for pid in "${!PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                END_TIME=$(date +%s.%N)
                START_TIME=${START_TIMES[$pid]}
                DURATION=$(echo "$END_TIME - $START_TIME" | bc)
                NP=${NP_FOR_PID[$pid]}
                echo "✅ Finished: np=$NP in ${DURATION}s"
                unset PIDS[$pid]
                unset CORES_USED[$pid]
                unset START_TIMES[$pid]
                unset NP_FOR_PID[$pid]
            fi
        done

        for pid in "${!CORES_USED[@]}"; do
            used=$((used + CORES_USED[$pid]))
        done

        if (( used + needed <= TOTAL_CORES )); then
            return
        fi
        sleep 1
    done
}

run_benchmark() {
    local NP=$1
    echo "🚀 Starting: np=$NP using $NP cores"
    START_TIME=$(date +%s.%N)

    /usr/bin/time -f "NP=$NP Time=%e sec CPU=%P MaxMem=%M KB Exit=%x" \
        mpirun -np "$NP" python3 parallel_karman.py \
        &>> "$LOG_FILE" &

    local pid=$!
    PIDS[$pid]=1
    CORES_USED[$pid]=$NP
    START_TIMES[$pid]=$START_TIME
    NP_FOR_PID[$pid]=$NP
}

for NP in "${PROCESS_COUNTS[@]}"; do
    wait_for_cores "$NP"
    run_benchmark "$NP"
done

# Wait for remaining jobs and log their completion
while [ "${#PIDS[@]}" -gt 0 ]; do
    wait_for_cores 0
done

echo "🎉 All benchmarks completed. Results saved in $LOG_FILE"
