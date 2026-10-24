# Client image for bench.py, run inside the server's network namespace so
# queries go over loopback instead of Docker Desktop's port forwarding.
FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.5.4 qdrant-client==1.20.0 weaviate-client==4.24.0
