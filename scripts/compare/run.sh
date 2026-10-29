#!/usr/bin/env bash
# Runs bench.py against Qdrant and Weaviate in Docker, one server at a time,
# each with f32 and int8 vectors. Results land in target/compare/*.json.
#
#   scripts/compare/run.sh [extra bench.py args, e.g. --base 20000]
#
# Pinecone is not started here: export PINECONE_API_KEY and run
#   uv run scripts/compare/bench.py pinecone --quant f32
set -euo pipefail
cd "$(dirname "$0")/../.."

QDRANT_IMAGE=qdrant/qdrant:v1.20.0
WEAVIATE_IMAGE=cr.weaviate.io/semitechnologies/weaviate:1.40.2
OUT=target/compare
mkdir -p "$OUT"
trap 'docker rm -f bench-qdrant bench-weaviate >/dev/null 2>&1 || true' EXIT

wait_http() {
    for _ in $(seq 1 120); do curl -sf "$1" >/dev/null && return 0; sleep 1; done
    echo "server at $1 never came up" >&2; exit 1
}

start_qdrant() {
    docker run -d --rm --name bench-qdrant -p 6333:6333 -p 6334:6334 \
        -e QDRANT__TELEMETRY_DISABLED=true "$QDRANT_IMAGE" >/dev/null
    wait_http http://localhost:6333/readyz
}

start_weaviate() {
    docker run -d --rm --name bench-weaviate -p 8080:8080 -p 50051:50051 \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
        -e DEFAULT_VECTORIZER_MODULE=none \
        -e CLUSTER_HOSTNAME=node1 \
        -e DISABLE_TELEMETRY=true \
        "$WEAVIATE_IMAGE" >/dev/null
    wait_http http://localhost:8080/v1/.well-known/ready
}

docker build -q -t vdblite-bench-client scripts/compare >/dev/null

for engine in qdrant weaviate; do
    for quant in f32 int8; do
        # A fresh container per run, so memory is that configuration's alone.
        "start_$engine"
        echo "== $engine $quant ($(docker inspect -f '{{.Config.Image}}' "bench-$engine"))"
        docker run --rm --network "container:bench-$engine" \
            -v "$PWD/data/sift:/data:ro" -v "$PWD/scripts/compare:/bench:ro" -v "$PWD/$OUT:/out" \
            vdblite-bench-client python /bench/bench.py "$engine" --quant "$quant" \
            --data /data --out "/out/$engine-$quant.json" "$@" \
            || status=$?
        if [[ -z "${status:-}" ]]; then
            mem=$(docker stats --no-stream --format '{{.MemUsage}}' "bench-$engine" | cut -d/ -f1 | xargs)
            case $engine in qdrant) dir=/qdrant/storage ;; weaviate) dir=/var/lib/weaviate ;; esac
            disk=$(docker exec "bench-$engine" du -sh "$dir" | cut -f1)
            echo "container memory $mem, data on disk $disk"
            python3 -c 'import json,sys; p,m,d=sys.argv[1:]; r=json.load(open(p)); r.update(memory=m, disk=d); json.dump(r, open(p,"w"), indent=2)' \
                "$OUT/$engine-$quant.json" "$mem" "$disk"
        fi
        # rm -f, not stop: stop returns before --rm finishes removing the
        # container, and the next run with the same name then conflicts.
        docker rm -f "bench-$engine" >/dev/null
        [[ -z "${status:-}" ]] || exit "$status"
    done
done
