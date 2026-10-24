#!/usr/bin/env bash
# Embedded engines head to head: vecfile, Zvec, LanceDB and sqlite-vec, each
# in the same Linux container image, one at a time. Per engine, a build
# container loads and indexes the data; then a fresh search container drops
# the page cache (hence --privileged) and measures. Databases live on a
# Docker volume inside the VM, not on a macOS bind mount.
#
#   uv run scripts/compare/embedded/prep.py sift            # once per dataset
#   scripts/compare/embedded/run.sh sift-1000000 [engine ...]
#   NO_BUILD=1 scripts/compare/embedded/run.sh arxiv768-500000   # same image again
#
# Engines: vecfile zvec lancedb-hnsw lancedb-ivfpq sqlite-vec (default: all).
# Results: target/compare/embedded/<dataset>/<engine>-{build,search}.json.
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATASET=${1:?usage: run.sh DATASET [engine ...]}
shift
ENGINES=("${@:-vecfile zvec lancedb-hnsw lancedb-ivfpq sqlite-vec}")
read -r -a ENGINES <<<"${ENGINES[*]}"
[[ -f data/bench/$DATASET/meta.json ]] || { echo "no data/bench/$DATASET; run prep.py" >&2; exit 1; }

IMAGE=vecfile-embedded-bench
VOLUME=bench-embedded-db
OUT=target/compare/embedded/$DATASET
mkdir -p "$OUT"
trap 'docker rm -f bench-embedded >/dev/null 2>&1 || true' EXIT

# NO_BUILD=1 reuses the existing image, so several datasets run the same build.
[[ ${NO_BUILD:-} == 1 ]] || docker build -q -t "$IMAGE" -f scripts/compare/embedded/Dockerfile . >/dev/null
docker volume create "$VOLUME" >/dev/null

run() { # run <extra docker args...> -- <command...>
    local args=()
    while [[ $1 != -- ]]; do args+=("$1"); shift; done
    shift
    docker run --rm --name bench-embedded -e BENCH_QUERIES ${args[@]+"${args[@]}"} \
        -v "$PWD/data:/data:ro" -v "$PWD/scripts/compare/embedded:/bench:ro" -v "$VOLUME:/db" -v "$PWD/$OUT:/out" "$IMAGE" "$@"
}

for engine in "${ENGINES[@]}"; do
    case $engine in
        vecfile) db=/db/vecfile.vdb ;;
        zvec | lancedb-hnsw | lancedb-ivfpq | sqlite-vec) db=/db/$engine ;;
        *) echo "unknown engine $engine" >&2; exit 1 ;;
    esac
    echo "== $engine on $DATASET"
    run -- sh -c "rm -rf $db"
    if [[ $engine == vecfile ]]; then
        run -- ann_compare build --dataset "$DATASET" --db "$db" --out "/out/$engine-build.json"
        search=(ann_compare search)
    else
        run -- python /bench/bench.py build "$engine" --dataset "$DATASET" --db "$db" --out "/out/$engine-build.json"
        search=(python /bench/bench.py search "$engine")
    fi
    run --privileged -- sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches && exec "$@"' sh \
        "${search[@]}" --dataset "$DATASET" --db "$db" --out "/out/$engine-search.json"
    run -- sh -c "rm -rf $db"
done
