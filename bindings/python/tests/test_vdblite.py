"""Smoke tests for the Python bindings: run with pytest after `maturin develop`."""

import random

import pytest
import vdblite


def unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def test_write_search_reopen(tmp_path):
    path = tmp_path / "t.vdb"
    rng = random.Random(7)
    vectors = [unit([rng.uniform(-1, 1) for _ in range(16)]) for _ in range(500)]
    keys = [str(i) for i in range(len(vectors))]

    db = vdblite.Db(path)
    coll = db.create_collection("v", 16, metric="cosine", seal_threshold=128)
    assert coll.upsert_vectors(keys, vectors) == 500
    db.optimize()
    hits = coll.search(vectors[42], k=5)
    assert hits[0][0] == "42"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert coll.index_stats()["chunks"] == 500
    del coll, db

    reader = vdblite.Db(path, read_only=True)
    assert reader.collections() == ["v"]
    assert reader.collection("v").search(vectors[7], k=1, ef=100)[0][0] == "7"
    with pytest.raises(vdblite.VdbliteError):
        reader.collection("v").upsert_vectors(["x"], [vectors[0]])


def test_errors(tmp_path):
    db = vdblite.Db(tmp_path / "e.vdb")
    with pytest.raises(ValueError):
        db.create_collection("bad", 4, metric="hamming")
    coll = db.create_collection("v", 4)
    with pytest.raises(vdblite.VdbliteError):
        coll.upsert_vectors(["a"], [[1.0, 2.0]])
    with pytest.raises(ValueError):
        coll.upsert_vectors(["a", "b"], [[1.0, 0, 0, 0]])
    with pytest.raises(vdblite.VdbliteError):
        db.collection("missing")


def test_bulk_load(tmp_path):
    rng = random.Random(3)
    rows = [unit([rng.uniform(-1, 1) for _ in range(16)]) for _ in range(3000)]
    db = vdblite.Db(tmp_path / "b.vdb")
    coll = db.create_collection("v", 16)
    load = coll.bulk_load()
    for s in range(0, 3000, 1000):
        keys = [str(i) for i in range(s, s + 1000)]
        assert load.add(keys, rows[s : s + 1000]) == 1000
    assert coll.search(rows[5], k=1) == [], "not searchable before finish"
    load.finish()
    with pytest.raises(vdblite.VdbliteError):
        load.finish()
    assert coll.search(rows[5], k=1)[0][0] == "5"
    stats = coll.index_stats()
    assert (stats["chunks"], stats["sealed_segments"], stats["buffered"]) == (3000, 1, 0)
