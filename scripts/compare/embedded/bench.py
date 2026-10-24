"""In-process vector search engines on a prepared dataset (see prep.py),
measured the way ann_compare.rs measures vecfile. run.sh runs each phase in
its own container:

    bench.py build  <engine> --dataset DIR --db PATH --out build.json
    bench.py search <engine> --dataset DIR --db PATH --out search.json

build: bulk-load the base vectors in 10K batches, build the index, close.
search: open in a fresh process (run.sh drops the page cache first), one
cold pass, then a warm sweep over each engine's accuracy knob: recall@10 by
id against exact ground truth, single-query latency, resident memory, and
concurrent throughput.

Engines: zvec, lancedb-hnsw (HNSW-SQ in one partition, configured like
vecfile), lancedb-ivfpq (LanceDB's default index), sqlite-vec (exact scan).
"""

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# LanceDB logs a deprecation warning on every query that selects columns.
os.environ.setdefault("LANCEDB_LOG", "error")
os.environ.setdefault("RUST_LOG", "error")

K = 10
BATCH = 10_000
M, EF_CONSTRUCTION = 16, 200  # vecfile's defaults
RERANK = 4  # vecfile re-ranks k x 4 candidates with f32
# vecfile raises ef to at least k x 4 = 40 (and LanceDB requires ef >= k x
# refine), so every HNSW sweep starts there.
EFS = [40, 64, 128, 256, 512]
REF_EF = 64  # operating point for the cold pass and throughput


def rows(path, limit=None):
    dim = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    raw = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, dim + 1)
    return raw[:limit, 1:].view(np.float32)


def rss_mb(field="VmRSS"):
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    return float("nan")


def du_mb(path):
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


# ---------------------------------------------------------------- engines


class Zvec:
    knobs = [{"ef": ef} for ef in EFS]
    ref = {"ef": REF_EF}

    def __init__(self, dim, metric):
        import zvec

        self.z, self.dim = zvec, dim
        self.metric = {"l2": zvec.MetricType.L2, "cosine": zvec.MetricType.COSINE}[metric]

    def create(self, path):
        z = self.z
        index = z.HnswIndexParam(
            metric_type=self.metric,
            m=M,
            ef_construction=EF_CONSTRUCTION,
            quantize_type=z.QuantizeType.INT8,
        )
        schema = z.CollectionSchema(
            name="bench", vectors=z.VectorSchema("v", z.DataType.VECTOR_FP32, self.dim, index_param=index)
        )
        self.c = z.create_and_open(path, schema)

    def add(self, start, vecs):
        Doc = self.z.Doc
        for s in range(0, len(vecs), 1024):  # zvec's maximum write batch
            statuses = self.c.insert(
                [Doc(id=str(start + i), vectors={"v": vecs[i]}) for i in range(s, min(s + 1024, len(vecs)))]
            )
            bad = [st for st in statuses if not st.ok()]
            if bad:
                raise SystemExit(f"zvec insert failed: {bad[0]}")

    def finish(self):
        self.c.optimize()

    def open(self, path):
        self.c = self.z.open(path)

    def set(self, ef):
        self.param = self.z.HnswQueryParam(ef=ef, is_using_refiner=True, scale_factor=float(RERANK))

    def search(self, q):
        res = self.c.query(self.z.Query("v", vector=q, param=self.param), topk=K, output_fields=[])
        return [int(d.id) for d in res]

    def close(self):
        self.c.close()


class LanceHnsw:
    knobs = [{"ef": ef} for ef in EFS]
    ref = {"ef": REF_EF}

    def __init__(self, dim, metric):
        import lancedb
        import pyarrow as pa

        self.lancedb, self.pa, self.dim, self.metric = lancedb, pa, dim, metric
        self.schema = pa.schema([pa.field("id", pa.int64()), pa.field("vector", pa.list_(pa.float32(), dim))])

    def index_config(self):
        from lancedb.index import HnswSq

        # One partition is one HNSW graph over everything, as in vecfile;
        # the LanceDB docs recommend it for HNSW indexes.
        return HnswSq(distance_type=self.metric, num_partitions=1, m=M, ef_construction=EF_CONSTRUCTION)

    def create(self, path):
        self.db = self.lancedb.connect(path)
        self.pending = []

    def add(self, start, vecs):
        pa = self.pa
        flat = pa.array(np.ascontiguousarray(vecs).reshape(-1))
        self.pending.append(
            pa.RecordBatch.from_arrays(
                [pa.array(np.arange(start, start + len(vecs), dtype=np.int64)),
                 pa.FixedSizeListArray.from_arrays(flat, self.dim)],
                schema=self.schema,
            )
        )

    def finish(self):
        # One bulk write (LanceDB's recommended load path), then the index.
        self.t = self.db.create_table("bench", data=self.pa.Table.from_batches(self.pending, self.schema))
        self.pending = None
        self.t.create_index("vector", config=self.index_config())

    def open(self, path):
        self.t = self.lancedb.connect(path).open_table("bench")

    def set(self, **knob):
        self.knob = knob

    def query(self, q):
        return self.t.search(q).distance_type(self.metric).limit(K).select(["id"])

    def search(self, q):
        b = self.query(q).nprobes(1).ef(self.knob["ef"]).refine_factor(RERANK)
        return b.to_arrow()["id"].to_pylist()

    def close(self):
        pass


class LanceIvfPq(LanceHnsw):
    knobs = [{"nprobes": n, "refine": r} for n, r in
             [(10, 4), (20, 4), (50, 4), (20, 10), (50, 10), (100, 10), (200, 10)]]
    ref = {"nprobes": 20, "refine": 10}

    def index_config(self):
        from lancedb.index import IvfPq

        return IvfPq(distance_type=self.metric)

    def search(self, q):
        b = self.query(q).nprobes(self.knob["nprobes"]).refine_factor(self.knob["refine"])
        return b.to_arrow()["id"].to_pylist()


class SqliteVec:
    knobs = [{}]  # exact scan: one operating point
    ref = {}

    def __init__(self, dim, metric):
        import sqlite3

        import sqlite_vec

        self.sqlite3, self.sqlite_vec, self.dim, self.metric = sqlite3, sqlite_vec, dim, metric
        self.local = threading.local()

    def connect(self, path):
        con = self.sqlite3.connect(path, check_same_thread=False)
        con.enable_load_extension(True)
        self.sqlite_vec.load(con)
        con.enable_load_extension(False)
        return con

    def create(self, path):
        self.path = path
        self.con = self.connect(path)
        self.con.execute(
            f"create virtual table v using vec0(embedding float[{self.dim}] distance_metric={self.metric})"
        )

    def add(self, start, vecs):
        # rowid = id + 1: SQLite rowids start at 1.
        with self.con:
            self.con.executemany(
                "insert into v(rowid, embedding) values (?, ?)",
                ((start + i + 1, v.tobytes()) for i, v in enumerate(vecs)),
            )

    def finish(self):
        pass

    def open(self, path):
        self.path = path
        self.con = self.connect(path)

    def set(self, **knob):
        pass

    def search(self, q):
        # One connection per thread, as SQLite expects.
        con = getattr(self.local, "con", None)
        if con is None:
            con = self.local.con = self.con if threading.current_thread() is threading.main_thread() else self.connect(self.path)
        res = con.execute("select rowid from v where embedding match ? and k = ?", (q.tobytes(), K))
        return [r[0] - 1 for r in res]

    def close(self):
        self.con.close()


ENGINES = {"zvec": Zvec, "lancedb-hnsw": LanceHnsw, "lancedb-ivfpq": LanceIvfPq, "sqlite-vec": SqliteVec}


# ---------------------------------------------------------------- measurement


def recall(found, gt):
    return sum(len(set(f[:K]) & set(t[:K])) for f, t in zip(found, gt)) / (len(found) * K)


def pct(lat, p):
    return lat[min(int(len(lat) * p), len(lat) - 1)]


def timed_pass(eng, queries):
    lat, found = [], []
    for q in queries:
        t = time.perf_counter()
        found.append(eng.search(q))
        lat.append((time.perf_counter() - t) * 1e6)
    return sorted(lat), found


def summary(lat, found, gt):
    return {"recall": recall(found, gt), "p50_us": pct(lat, 0.5), "p99_us": pct(lat, 0.99),
            "mean_us": sum(lat) / len(lat)}


def concurrent_qps(eng, queries, threads, per_thread):
    barrier = threading.Barrier(threads + 1)

    def run(th):
        barrier.wait()
        for i in range(per_thread):
            eng.search(queries[(th * 97 + i) % len(queries)])

    ts = [threading.Thread(target=run, args=(th,)) for th in range(threads)]
    for t in ts:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    return threads * per_thread / (time.perf_counter() - t0)


def build(a, meta, eng_cls):
    base = rows(f"{a.data}/{meta['base']}", meta["n"])
    eng = eng_cls(meta["dim"], meta["metric"])
    rss0 = rss_mb()
    t = time.perf_counter()
    eng.create(a.db)
    for s in range(0, len(base), BATCH):
        eng.add(s, np.ascontiguousarray(base[s : s + BATCH]))
    loaded = time.perf_counter() - t
    eng.finish()
    total = time.perf_counter() - t
    eng.close()
    res = {"phase": "build", "load_s": loaded, "index_s": total - loaded, "ingest_s": total,
           "disk_mb": du_mb(a.db), "peak_rss_mb": rss_mb("VmHWM"), "rss_before_mb": rss0}
    print(f"ingest {meta['n']} in {total:.1f} s (load {loaded:.1f} s, index {total - loaded:.1f} s), "
          f"disk {res['disk_mb']:.0f} MB, peak RSS {res['peak_rss_mb']:.0f} MB")
    return res


def floor_us(eng_cls, meta, queries):
    """Same query path on a 10-vector collection: binding and per-query
    overhead with almost no search."""
    tmp = tempfile.mkdtemp()
    eng = eng_cls(meta["dim"], meta["metric"])
    eng.create(os.path.join(tmp, "tiny"))
    eng.add(0, np.ascontiguousarray(queries[:10]))
    if isinstance(eng, LanceHnsw):
        # Too few rows to train an index: time the plain query path.
        eng.t = eng.db.create_table("bench", data=eng.pa.Table.from_batches(eng.pending, eng.schema))
        search = lambda q: eng.query(q).to_arrow()["id"].to_pylist()  # noqa: E731
    else:
        eng.finish()
        set_knob(eng, eng.ref)
        search = eng.search
    for q in queries[:300]:
        search(q)
    lat = []
    for q in queries:
        t = time.perf_counter()
        search(q)
        lat.append((time.perf_counter() - t) * 1e6)
    eng.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return pct(sorted(lat), 0.5)


def set_knob(eng, knob):
    if isinstance(eng, Zvec):
        eng.set(knob["ef"])
    else:
        eng.set(**knob)


def search(a, meta, eng_cls):
    queries = rows(f"{a.data}/bench/{meta['name']}/query.fvecs")
    queries = np.ascontiguousarray(queries)
    gt = rows(f"{a.data}/bench/{meta['name']}/gt.ivecs").view(np.int32)[:, :K].tolist()
    # BENCH_QUERIES=N times only the first N queries (for the exact scan,
    # whose passes over 1,000 queries take tens of minutes at 768 dims).
    limit = int(os.environ.get("BENCH_QUERIES") or len(queries))
    queries, gt = queries[:limit], gt[:limit]
    eng = eng_cls(meta["dim"], meta["metric"])
    rss0 = rss_mb()
    t = time.perf_counter()
    eng.open(a.db)
    res = {"phase": "search", "queries": len(queries), "open_s": time.perf_counter() - t, "rss_open_mb": rss_mb() - rss0}

    set_knob(eng, eng.ref)
    t = time.perf_counter()
    eng.search(queries[0])
    res["first_query_us"] = (time.perf_counter() - t) * 1e6
    lat, found = timed_pass(eng, queries)
    res["cold"] = {"knob": eng.ref, **summary(lat, found, gt)}
    print(f"open {res['open_s'] * 1e3:.0f} ms, first query {res['first_query_us'] / 1e3:.1f} ms, "
          f"cold pass p50 {res['cold']['p50_us']:.0f} µs p99 {res['cold']['p99_us']:.0f} µs")

    res["warm"] = []
    for knob in eng.knobs:
        set_knob(eng, knob)
        timed_pass(eng, queries)  # warm-up
        lat, found = timed_pass(eng, queries)
        s = {"knob": knob, **summary(lat, found, gt)}
        res["warm"].append(s)
        print(f"warm {knob}: recall@10 {s['recall']:.4f}  p50 {s['p50_us']:.0f} µs  p99 {s['p99_us']:.0f} µs")
    res["rss_mb"] = rss_mb() - rss0

    set_knob(eng, eng.ref)
    threads = os.cpu_count() or 1
    # Enough work for a few seconds; the exact scan needs far fewer queries.
    per = 40 if isinstance(eng, SqliteVec) else 600
    res["qps"] = {"threads": threads, "knob": eng.ref, "qps": concurrent_qps(eng, queries, threads, per)}
    print(f"concurrent {eng.ref}, {threads} threads: {res['qps']['qps']:.0f} QPS")
    eng.close()

    res["floor_p50_us"] = floor_us(eng_cls, meta, queries)
    print(f"per-query floor (10-vector collection): p50 {res['floor_p50_us']:.0f} µs; resident {res['rss_mb']:.0f} MB")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["build", "search"])
    p.add_argument("engine", choices=ENGINES)
    p.add_argument("--data", default="/data", help="the repo's data/ directory")
    p.add_argument("--dataset", required=True, help="prepared dataset name, e.g. sift-1000000")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    meta = json.loads(Path(f"{a.data}/bench/{a.dataset}/meta.json").read_text())
    eng_cls = ENGINES[a.engine]
    res = (build if a.phase == "build" else search)(a, meta, eng_cls)
    import importlib.metadata as md

    pkg = {"zvec": "zvec", "sqlite-vec": "sqlite-vec"}.get(a.engine, "lancedb")
    res.update(engine=a.engine, version=md.version(pkg), dataset=meta)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
