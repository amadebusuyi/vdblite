#!/usr/bin/env bash
# Downloads real 768-dim text embeddings into data/arxiv768: InstructorXL
# embeddings of arXiv paper titles, published by Qdrant on Hugging Face
# (Qdrant/arxiv-titles-instructorxl-embeddings), 50K rows per part.
#
#   scripts/fetch-arxiv768.sh [PARTS]   # default 20 parts = 1M vectors (~3.2 GB download)
#
# Writes base.fvecs (vectors, in file order) and titles.txt (one title per
# line, same order). Converting the parquet needs `uv` (it fetches pyarrow
# into a throwaway environment). Safe to re-run.
set -euo pipefail

PARTS="${1:-20}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/arxiv768"
URL="https://huggingface.co/datasets/Qdrant/arxiv-titles-instructorxl-embeddings/resolve/main/data"

mkdir -p "$DEST/raw"
expected=$((PARTS * 50000))
if [[ -f "$DEST/base.fvecs" ]]; then
    have=$(( $(wc -c < "$DEST/base.fvecs" | tr -d ' ') / (4 + 768 * 4) ))
    if (( have >= expected )); then
        echo "arxiv768 already has $have vectors in $DEST"
        exit 0
    fi
fi

for ((i = 0; i < PARTS; i++)); do
    part=$(printf "part-%02d-arxiv-titles-instructorxl.parquet" "$i")
    if [[ ! -f "$DEST/raw/$part" ]]; then
        echo "Downloading $part (~160 MB)..."
        curl -sSL --retry 3 -o "$DEST/raw/$part.tmp" "$URL/$part"
        mv "$DEST/raw/$part.tmp" "$DEST/raw/$part"
    fi
done

echo "Converting to fvecs..."
uv run -q --no-project --with pyarrow python -I "$ROOT/scripts/arxiv768.py" "$DEST" "$PARTS"
rm -rf "$DEST/raw"
echo "arxiv768 ready in $DEST"
