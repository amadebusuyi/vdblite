# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.5.4",
#   "qdrant-client==1.20.0",
#   "weaviate-client==4.24.0",
#   "pinecone[grpc]==10.2.0",
# ]
# ///
"""SIFT1M against a vector database server, measured the way sift_db.rs
measures vdblite: the first 1,000 queries, k=10, recall@10 by id against the
published ground truth, single-query latency over a warm pass, then
concurrent throughput at ef=64.

    uv run scripts/compare/bench.py qdrant --quant int8 [--base N] [--host H]

Engines: qdrant, weaviate (run.sh starts them and runs this script inside the
server's network namespace), pinecone (needs PINECONE_API_KEY; serverless, so
latency includes the internet).
Results print as text and are written as JSON to --out.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import uuid

import numpy as np

K = 10
EFS = (32, 64, 128)
DIM = 128


def read_vecs(path, dtype, limit=None):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    rows = raw.reshape(-1, dim + 1)[:, 1:]
    if limit is not None:
        rows = rows[:limit]
    return np.ascontiguousarray(rows.view(dtype))


# ---------------------------------------------------------------- engines


class Qdrant:
    def __init__(self, a):
        from qdrant_client import QdrantClient, models

        self.m = models
        self.a = a
        self.c = QdrantClient(host=a.host, grpc_port=6334, prefer_grpc=True, timeout=3600)

    def create(self, name, n_hint):
        m = self.m
        if self.c.collection_exists(name):
            self.c.delete_collection(name)
        quant = None
        if self.a.quant == "int8":
            quant = m.ScalarQuantization(
                scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8, always_ram=True)
            )
        self.c.create_collection(
            name,
            vectors_config=m.VectorParams(size=DIM, distance=m.Distance.EUCLID),
            hnsw_config=m.HnswConfigDiff(m=16, ef_construct=200),
            quantization_config=quant,
            # Bulk-load recipe from the Qdrant docs: no indexing during
            # upload, then build the graphs once.
            optimizers_config=m.OptimizersConfigDiff(indexing_threshold=0),
        )

    def ingest(self, name, vectors):
        self.c.upload_collection(
            name, vectors=vectors, ids=range(len(vectors)), batch_size=1000, parallel=8
        )
        self.c.update_collection(
            name, optimizers_config=self.m.OptimizersConfigDiff(indexing_threshold=10000)
        )

    def wait_ready(self, name, n):
        # GREEN alone can be read before the optimizer picks up the new
        # threshold; also wait for the graphs to cover the data. Collections
        # under the threshold (the floor) are never indexed.
        need = int(0.9 * n) if n >= 100_000 else 0
        while True:
            info = self.c.get_collection(name)
            if (
                info.status == self.m.CollectionStatus.GREEN
                and (info.indexed_vectors_count or 0) >= need
            ):
                return info.segments_count
            time.sleep(1)

    def set_ef(self, name, ef):
        self.ef = ef

    def search(self, name, q):
        m = self.m
        params = m.SearchParams(hnsw_ef=self.ef)
        if self.a.quant == "int8":
            # Re-rank 4 x k candidates with f32, as vdblite does.
            params.quantization = m.QuantizationSearchParams(rescore=True, oversampling=4.0)
        res = self.c.query_points(
            name, query=q.tolist(), limit=K, search_params=params, with_payload=False
        )
        return [p.id for p in res.points]

    def close(self):
        self.c.close()


class Weaviate:
    def __init__(self, a):
        import weaviate

        self.a = a
        self.c = weaviate.connect_to_local(host=a.host, skip_init_checks=True)

    def create(self, name, n_hint):
        from weaviate.classes.config import Configure, VectorDistances

        if self.c.collections.exists(name):
            self.c.collections.delete(name)
        if self.a.quant == "int8":
            quantizer = Configure.VectorIndex.Quantizer.sq(rescore_limit=4 * K)
        else:
            # Explicit, since recent Weaviate compresses new collections by default.
            quantizer = Configure.VectorIndex.Quantizer.none()
        self.c.collections.create(
            name,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.L2_SQUARED,
                    max_connections=16,
                    ef_construction=200,
                    ef=64,
                    quantizer=quantizer,
                )
            ),
        )

    def ingest(self, name, vectors):
        col = self.c.collections.use(name)
        with col.batch.fixed_size(batch_size=1000, concurrent_requests=8) as b:
            for i, v in enumerate(vectors):
                b.add_object(properties={}, uuid=uuid.UUID(int=i), vector=v.tolist())
        failed = col.batch.failed_objects
        if failed:
            sys.exit(f"weaviate: {len(failed)} objects failed, first: {failed[0].message}")

    def wait_ready(self, name, n):
        col = self.c.collections.use(name)
        while True:
            shards = col.config.get_shards()
            if all(s.status == "READY" and s.vector_queue_size == 0 for s in shards):
                return len(shards)
            time.sleep(1)

    def set_ef(self, name, ef):
        from weaviate.classes.config import Reconfigure

        col = self.c.collections.use(name)
        col.config.update(
            vector_config=Reconfigure.Vectors.update(
                name="default", vector_index_config=Reconfigure.VectorIndex.hnsw(ef=ef)
            )
        )
        # The change is applied asynchronously; wait until it reads back.
        while col.config.get().vector_config["default"].vector_index_config.ef != ef:
            time.sleep(0.2)
        time.sleep(1)

    def search(self, name, q):
        col = self.c.collections.use(name)
        res = col.query.near_vector(near_vector=q.tolist(), limit=K, return_properties=[])
        return [o.uuid.int for o in res.objects]

    def close(self):
        self.c.close()


class Pinecone:
    """Serverless index. No HNSW knobs and no ef: one operating point."""

    def __init__(self, a):
        from pinecone.grpc import PineconeGRPC

        self.a = a
        self.pc = PineconeGRPC(api_key=os.environ["PINECONE_API_KEY"])
        self.idx = {}

    def index(self, name):
        if name not in self.idx:
            self.idx[name] = self.pc.Index(host=self.pc.describe_index(name).host)
        return self.idx[name]

    def create(self, name, n_hint):
        from pinecone import ServerlessSpec

        if self.pc.has_index(name):
            self.pc.delete_index(name)
        self.pc.create_index(
            name=name,
            dimension=DIM,
            metric="euclidean",
            spec=ServerlessSpec(cloud=self.a.pinecone_cloud, region=self.a.pinecone_region),
        )
        while not self.pc.describe_index(name).status["ready"]:
            time.sleep(1)

    def ingest(self, name, vectors):
        idx = self.index(name)
        step = 50_000  # the client batches and parallelises within each call
        for s in range(0, len(vectors), step):
            rows = range(s, min(s + step, len(vectors)))
            idx.upsert(
                vectors=[(str(i), vectors[i].tolist()) for i in rows],
                batch_size=500,
                show_progress=False,
            )

    def wait_ready(self, name, n):
        while self.index(name).describe_index_stats().total_vector_count < n:
            time.sleep(2)
        return None

    def set_ef(self, name, ef):
        pass

    def search(self, name, q):
        res = self.index(name).query(vector=q.tolist(), top_k=K)
        return [int(m.id) for m in res.matches]

    def close(self):
        pass


ENGINES = {"qdrant": Qdrant, "weaviate": Weaviate, "pinecone": Pinecone}


# ---------------------------------------------------------------- measurement


def recall(found, gt):
    return sum(len(set(f[:K]) & set(t[:K])) for f, t in zip(found, gt)) / (len(found) * K)


def pct(lat, p):
    return lat[min(int(len(lat) * p), len(lat) - 1)]


def timed_pass(eng, name, queries):
    lat, found = [], []
    for q in queries:
        t = time.perf_counter()
        found.append(eng.search(name, q))
        lat.append((time.perf_counter() - t) * 1e6)
    lat.sort()
    return lat, found


def worker(a, name, queries, offset, count, barrier, out):
    eng = ENGINES[a.engine](a)
    eng.ef = 64  # Qdrant takes ef per query; the others were set by the parent
    for q in queries[:20]:
        eng.search(name, q)
    barrier.wait()
    t = time.perf_counter()
    for i in range(count):
        eng.search(name, queries[(offset + i) % len(queries)])
    out.put((t, time.perf_counter()))
    eng.close()


def concurrent_qps(a, name, queries, clients, total):
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(clients), ctx.Queue()
    per = total // clients
    procs = [
        ctx.Process(target=worker, args=(a, name, queries, c * 97, per, barrier, out))
        for c in range(clients)
    ]
    for p in procs:
        p.start()
    spans = [out.get() for _ in procs]
    for p in procs:
        p.join()
    wall = max(e for _, e in spans) - min(s for s, _ in spans)
    return per * clients / wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("engine", choices=ENGINES)
    p.add_argument("--quant", choices=["f32", "int8"], default="int8")
    p.add_argument("--data", default="data/sift")
    p.add_argument("--base", type=int, help="ingest only the first N base vectors (no recall)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--reuse", action="store_true", help="search an already-loaded collection")
    p.add_argument("--clients", default="4,8,16,32")
    p.add_argument("--out")
    p.add_argument("--pinecone-cloud", default="aws")
    p.add_argument("--pinecone-region", default="us-east-1")
    a = p.parse_args()

    base = read_vecs(f"{a.data}/sift_base.fvecs", np.float32, a.base)
    queries = read_vecs(f"{a.data}/sift_query.fvecs", np.float32, 1000)
    gt = read_vecs(f"{a.data}/sift_groundtruth.ivecs", np.int32, 1000).tolist()
    n = len(base)
    full = n == 1_000_000
    name = f"sift{a.quant}" if a.engine != "pinecone" else f"sift-{a.quant}"
    res = {"engine": a.engine, "quant": a.quant, "n": n}

    eng = ENGINES[a.engine](a)

    # Round-trip floor: the same query path against a 10-vector collection,
    # i.e. client, transport and request handling with almost no search.
    if a.engine != "pinecone":
        tiny = name + "floor"
        eng.create(tiny, 10)
        eng.ingest(tiny, base[:10])
        eng.wait_ready(tiny, 10)
        eng.set_ef(tiny, 64)
        timed_pass(eng, tiny, queries[:200])
        lat, _ = timed_pass(eng, tiny, queries)
        res["floor_p50_us"] = pct(lat, 0.5)
        print(f"round-trip floor (10-vector collection): p50 {pct(lat, 0.5):.0f} µs  p99 {pct(lat, 0.99):.0f} µs")

    if not a.reuse:
        eng.create(name, n)
        t = time.perf_counter()
        eng.ingest(name, base)
        upload = time.perf_counter() - t
        segments = eng.wait_ready(name, n)
        total = time.perf_counter() - t
        res.update(upload_s=upload, ingest_s=total, segments=segments)
        print(f"ingested {n} in {total:.1f} s ({n / total:.0f} vec/s; upload {upload:.1f} s, "
              f"then {total - upload:.1f} s until indexed); segments/shards: {segments}")

    res["searches"] = []
    for ef in EFS:
        eng.set_ef(name, ef)
        timed_pass(eng, name, queries)  # warm-up
        lat, found = timed_pass(eng, name, queries)
        r = recall(found, gt) if full else float("nan")
        res["searches"].append(
            {"ef": ef, "recall": r, "p50_us": pct(lat, 0.5), "p99_us": pct(lat, 0.99)}
        )
        print(f"warm ef={ef:<4} recall@10 (by id) {r:.4f}  p50 {pct(lat, 0.5):.0f} µs  p99 {pct(lat, 0.99):.0f} µs")
        if a.engine == "pinecone":
            break

    eng.set_ef(name, 64)
    res["concurrent"] = {}
    for c in map(int, a.clients.split(",")):
        qps = concurrent_qps(a, name, queries, c, 10_000)
        res["concurrent"][c] = qps
        print(f"concurrent ef=64, {c} client processes: {qps:.0f} QPS")

    eng.close()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
