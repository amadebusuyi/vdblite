#!/usr/bin/env bash
# Downloads SIFT1M (base, queries, ground truth) into data/sift. Safe to re-run.
set -euo pipefail

DEST="${1:-$(dirname "$0")/../data}"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

mkdir -p "$DEST"
cd "$DEST"

if [[ -f sift/sift_base.fvecs && -f sift/sift_query.fvecs && -f sift/sift_groundtruth.ivecs ]]; then
    echo "SIFT1M already present in $(pwd)/sift"
    exit 0
fi

echo "Downloading $URL (~160 MB)..."
curl -sS --retry 3 -o sift.tar.gz "$URL"
tar -xzf sift.tar.gz
rm sift.tar.gz

# Expected sizes: 1M x (4 + 128*4), 10K x (4 + 128*4), 10K x (4 + 100*4).
check() {
    local size
    size=$(wc -c < "sift/$1" | tr -d ' ')
    [[ "$size" == "$2" ]] || { echo "sift/$1: expected $2 bytes, got $size" >&2; exit 1; }
}
check sift_base.fvecs 516000000
check sift_query.fvecs 5160000
check sift_groundtruth.ivecs 4040000

echo "SIFT1M ready in $(pwd)/sift"
