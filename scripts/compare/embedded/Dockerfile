# One image for every engine in the embedded comparison, so all of them run
# on the same OS, kernel and CPU allocation. Build from the repo root:
#   docker build -f scripts/compare/embedded/Dockerfile .
FROM rust:1.99.1-slim-trixie AS vecfile
WORKDIR /src
COPY . .
RUN cargo build --release -p vecfile --example ann_compare \
    && cp target/release/examples/ann_compare /ann_compare

FROM python:3.13-slim-trixie
RUN pip install --no-cache-dir numpy==2.5.4 zvec==0.8.0 lancedb==0.40.0 sqlite-vec==0.1.9
COPY --from=vecfile /ann_compare /usr/local/bin/ann_compare
COPY scripts/compare/embedded/bench.py /bench/bench.py
