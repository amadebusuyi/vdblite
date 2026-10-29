#!/usr/bin/env bash
# vdblite and Zvec through VectorDBBench on Cohere 1M, each in its own
# container from the same image (scripts/vectordbbench/Dockerfile), one
# after the other, with the same graph settings (M=16, efConstruction=200,
# int8 codes and a full-precision re-rank):
#
#   docker build -f scripts/vectordbbench/Dockerfile -t vdbb-embedded .
#   scripts/vectordbbench/docker-run.sh [vdblite|zvec ...]
#
# Each engine gets a full run (load, then searches at 1-16 processes), then
# warm search-only passes at its default ef and at ef 800. Datasets come
# from data/vectordbbench/dataset (read-only); databases live on a Docker
# volume, inside the VM, so memory-mapped I/O doesn't go through the host
# file share. Logs go to data/vectordbbench/docker-<engine>.log, results to
# data/vectordbbench/results-docker. Each container ends by printing its
# cgroup's peak memory (page cache of its files included).
set -euo pipefail
cd "$(dirname "$0")/../.."
engines=("${@:-vdblite zvec}")
read -r -a engines <<< "${engines[*]}"

common=(--case-type Performance768D1M --num-concurrency 1,2,4,8,12,16)
for engine in "${engines[@]}"; do
    case $engine in
        vdblite)
            cmd=(vdblite --path /db/vdblite-cohere1m.vdb --quant int8)
            efs=(64 800) # vdblite raises ef to k x 4 = 400
            ;;
        zvec)
            cmd=(zvec --path /db/zvec-cohere1m --m 16 --ef-construction 200
                --quantize-type int8 --is-using-refiner)
            efs=(400 800)
            ;;
        *) echo "unknown engine $engine" >&2; exit 2 ;;
    esac
    name=bench-vdbb-$engine
    if docker ps -q --filter "name=^bench-" | grep -q .; then
        echo "another bench-* container is running; one benchmark at a time" >&2
        exit 1
    fi
    log=data/vectordbbench/docker-$engine.log
    mkdir -p data/vectordbbench/results-docker
    # The run, as one shell script inside the container.
    script="set -u
run() { echo \"== \$* \$(date +%T)\"; vectordbbench \"\$@\"; echo \"exit \$?\"; }
run ${cmd[*]} ${common[*]} --ef-search ${efs[0]} --db-label docker-$engine-full
for ef in ${efs[*]}; do
    run ${cmd[*]} ${common[*]} --skip-drop-old --skip-load --ef-search \$ef --db-label docker-$engine-ef\$ef
done
echo \"== memory.peak \$(cat /sys/fs/cgroup/memory.peak)\"
echo \"== files \$(du -sb /db/*$engine* | cut -f1) bytes\"
date '+== end %T'"
    echo "== $engine: $(date +%T), log $log"
    docker run --rm --name "$name" \
        -v "$PWD/data/vectordbbench/dataset:/data/dataset:ro" \
        -v "$PWD/data/vectordbbench/results-docker:/data/results" \
        -v vdbb-db:/db \
        --entrypoint bash vdbb-embedded -c "$script" > "$log" 2>&1
done
