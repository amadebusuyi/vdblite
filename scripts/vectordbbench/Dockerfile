# VectorDBBench with the vdblite and Zvec clients in one image, so both run
# on the same OS, kernel and CPU allocation. Build from the repo root:
#
#   docker build -f scripts/vectordbbench/Dockerfile -t vdbb-embedded .
#
# See scripts/vectordbbench/docker-run.sh for running it.
FROM rust:1.99.1-slim-trixie AS wheel
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /maturin && /maturin/bin/pip install --no-cache-dir maturin==1.15.0
WORKDIR /src
COPY . .
# The extension has its own workspace (and release profile: it must unwind
# on panic).
RUN /maturin/bin/maturin build --release -m bindings/python/Cargo.toml -o /wheels

FROM python:3.13-slim-trixie
ARG VDBB_COMMIT=139f8a3d38fee6d3e8bd62ffbd8ea652b6172108
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git init -q /vdbb && git -C /vdbb remote add origin https://github.com/zilliztech/VectorDBBench.git \
    && git -C /vdbb fetch -q --depth 1 origin "$VDBB_COMMIT" \
    && git -C /vdbb checkout -q "$VDBB_COMMIT"
COPY scripts/vectordbbench/vdblite /vdbb/vectordb_bench/backend/clients/vdblite
COPY scripts/vectordbbench/register.py /tmp/register.py
RUN python /tmp/register.py /vdbb
# The versions the native runs used.
RUN pip install --no-cache-dir -e /vdbb zvec==0.8.0 numpy==2.5.4 pyarrow==26.0.0 pandas==3.1.0
COPY --from=wheel /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    && python -c 'import vdblite, zvec, vectordb_bench; print("ready")'
ENV DATASET_LOCAL_DIR=/data/dataset RESULTS_LOCAL_DIR=/data/results LOG_LEVEL=INFO
WORKDIR /work
ENTRYPOINT ["vectordbbench"]
