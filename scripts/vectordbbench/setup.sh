#!/usr/bin/env bash
# Sets up VectorDBBench (github.com/zilliztech/VectorDBBench, pinned) with a
# vdblite client, in one virtualenv with vdblite's Python bindings.
#
#   scripts/vectordbbench/setup.sh            # into target/vectordbbench
#   target/vectordbbench/venv/bin/vectordbbench vdblite --path /tmp/vdbb/cohere1m.vdb \
#       --case-type Performance768D1M --num-concurrency 1,4,8,16
#
# vdblite copies its index into each process that opens the file, and
# VectorDBBench searches with one process per concurrent client, so keep
# --num-concurrency to about the core count (the default goes to 80).
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMIT=139f8a3d38fee6d3e8bd62ffbd8ea652b6172108 # 2026-10-08
DIR=${1:-target/vectordbbench}
SRC=$DIR/VectorDBBench
mkdir -p "$DIR"

if [[ ! -d $SRC/.git ]]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/zilliztech/VectorDBBench.git
fi
if [[ $(git -C "$SRC" rev-parse -q --verify HEAD || true) != "$COMMIT" ]]; then
    git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
    git -C "$SRC" checkout -q --force "$COMMIT"
fi

rm -rf "$SRC/vectordb_bench/backend/clients/vdblite"
cp -R scripts/vectordbbench/vdblite "$SRC/vectordb_bench/backend/clients/vdblite"
python3 scripts/vectordbbench/register.py "$SRC"

[[ -x $DIR/venv/bin/python ]] || uv venv -q --python 3.13 "$DIR/venv"
VIRTUAL_ENV=$PWD/$DIR/venv uv pip install -q -e "$SRC" maturin==1.15.0
# The extension builds in its own target dir: it must unwind on panic,
# unlike the workspace's release profile.
(cd bindings/python && VIRTUAL_ENV=$PWD/../../$DIR/venv CARGO_TARGET_DIR=../../target/python \
    ../../$DIR/venv/bin/maturin develop --release -q)
"$DIR/venv/bin/python" -c 'import vdblite, vectordb_bench; print("vdblite binding and VectorDBBench ready")'
