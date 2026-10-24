# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4"]
# ///
"""Prepares a dataset for the embedded comparison: query vectors, exact
top-100 ground truth, and meta.json naming the base file, row count and
metric. Every engine reads the same files, so they search the same data.

    uv run scripts/compare/embedded/prep.py sift      [--limit N]
    uv run scripts/compare/embedded/prep.py arxiv768  [--limit N]

sift: the published SIFT1M queries; the published ground truth when all 1M
base vectors are used, exact search otherwise. arxiv768: real InstructorXL
embeddings (cosine); the last 1,000 rows of the file are held out as
queries, and the base is the first N rows.

Writes data/bench/<name>-<N>/{query.fvecs, gt.ivecs, meta.json}.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
QUERIES = 1000
GT_K = 100


def rows(path, limit=None, start=0):
    """Rows [start, start + limit) of an .fvecs/.ivecs file, memory-mapped."""
    dim = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    raw = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, dim + 1)
    end = len(raw) if limit is None else min(len(raw), start + limit)
    return raw[start:end, 1:]


def write_vecs(path, a):
    a = np.ascontiguousarray(a)
    out = np.empty((a.shape[0], a.shape[1] + 1), dtype=np.int32)
    out[:, 0] = a.shape[1]
    out[:, 1:] = a.view(np.int32)
    out.tofile(path)


def exact_topk(base, queries, metric):
    """Exact top-GT_K ids, scanning the base in blocks so memory stays flat."""
    q = queries.astype(np.float32)
    best_d = np.full((len(q), GT_K), np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), GT_K), dtype=np.int64)
    step = 100_000
    for s in range(0, len(base), step):
        b = np.asarray(base[s : s + step]).view(np.float32)
        if metric == "l2":
            d = (b * b).sum(1)[None, :] - 2 * q @ b.T  # |q|^2 is the same per row
        else:  # cosine over unit vectors: rank by -dot
            d = -(q @ b.T)
        d_all = np.concatenate([best_d, d], axis=1)
        i_all = np.concatenate([best_i, np.arange(s, s + len(b))[None, :].repeat(len(q), 0)], axis=1)
        part = np.argpartition(d_all, GT_K - 1, axis=1)[:, :GT_K]
        best_d = np.take_along_axis(d_all, part, 1)
        best_i = np.take_along_axis(i_all, part, 1)
        print(f"  ground truth: {s + len(b)} / {len(base)} rows", flush=True)
    order = np.argsort(best_d, axis=1, kind="stable")
    return np.take_along_axis(best_i, order, 1).astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["sift", "arxiv768"])
    p.add_argument("--limit", type=int, help="base rows (default: all of SIFT1M, 500K of arxiv768)")
    a = p.parse_args()

    if a.dataset == "sift":
        base_file = DATA / "sift/sift_base.fvecs"
        n = a.limit or 1_000_000
        queries = np.asarray(rows(DATA / "sift/sift_query.fvecs", QUERIES)).view(np.float32)
        metric = "l2"
    else:
        base_file = DATA / "arxiv768/base.fvecs"
        total = rows(base_file).shape[0]
        n = a.limit or 500_000
        if n > total - QUERIES:
            raise SystemExit(f"--limit must leave {QUERIES} held-out rows (file has {total})")
        queries = np.asarray(rows(base_file, QUERIES, total - QUERIES)).view(np.float32)
        metric = "cosine"

    base = rows(base_file, n)
    n = len(base)
    out = DATA / "bench" / f"{a.dataset}-{n}"
    out.mkdir(parents=True, exist_ok=True)
    write_vecs(out / "query.fvecs", queries)

    if a.dataset == "sift" and n == 1_000_000:
        gt = np.asarray(rows(DATA / "sift/sift_groundtruth.ivecs", QUERIES))
    else:
        gt = exact_topk(base, queries, metric)
    write_vecs(out / "gt.ivecs", gt)

    meta = {
        "name": f"{a.dataset}-{n}",
        "base": str(base_file.relative_to(DATA)),
        "n": n,
        "dim": int(base.shape[1]),
        "metric": metric,
        "queries": len(queries),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {meta}")


if __name__ == "__main__":
    main()
