"""vdblite: an embedded, single-file vector database (no server).

Each VectorDBBench process opens the file itself: the insert and optimize
processes for writing (one at a time), search processes read-only alongside.
Inserts arrive from several threads sharing one handle, in batches of 100.
They go into a bulk load, `commit_every` rows per call; the insert process
finishes it (building the index once, with every core) as it closes, so
the build counts toward VectorDBBench's insert time.
"""

import logging
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import vdblite

from ..api import VectorDB
from .config import VdbliteConfig, VdbliteIndexConfig

log = logging.getLogger(__name__)


class Vdblite(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: VdbliteConfig,
        db_case_config: VdbliteIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "vdblite"
        self.path = db_config["path"]
        self.dim = dim
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.ef = db_case_config.search_param()["ef"]
        self.commit_every = db_case_config.commit_every
        self.read_only = False
        self.db = None
        self.coll = None
        self._lock = None
        self._bulk = None
        self._keys: list[str] = []
        self._vectors: list[list[float]] = []

        if drop_old:
            Path(self.path).unlink(missing_ok=True)
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            db = vdblite.Db(self.path)
            db.create_collection(collection_name, dim, **db_case_config.index_param())
            log.info(f"created {self.path}: {db_case_config.index_param()}")
            del db

    def __getstate__(self):
        # Copied into spawned processes before init(): no handles, no lock.
        state = self.__dict__.copy()
        state.update(db=None, coll=None, _lock=None, _bulk=None)
        return state

    @contextmanager
    def init(self):
        self.db = vdblite.Db(self.path, read_only=self.read_only)
        self.coll = self.db.collection(self.collection_name)
        self._lock = threading.Lock()
        try:
            yield
        finally:
            self._flush(0)
            if self._bulk is not None:
                self._bulk.finish()
                self._bulk = None
            # The file closes once its handles are gone.
            self.coll = None
            self.db = None
            self._lock = None

    def _flush(self, at_least: int):
        """Commits the buffer if it holds at least `at_least` rows."""
        with self._lock:
            if not self._keys or len(self._keys) < at_least:
                return
            keys, vectors = self._keys, self._vectors
            self._keys, self._vectors = [], []
            if self._bulk is None:
                self._bulk = self.coll.bulk_load()
            bulk = self._bulk
        # Outside the lock: vdblite runs one write transaction at a time.
        bulk.add(keys, np.asarray(vectors, dtype=np.float32))

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            with self._lock:
                self._keys.extend(str(i) for i in metadata)
                self._vectors.extend(embeddings)
            self._flush(self.commit_every)
        except Exception as e:
            log.warning(f"vdblite insert failed: {e}")
            return 0, e
        return len(metadata), None

    def search_embedding(self, query: list[float], k: int = 100, filters: dict | None = None) -> list[int]:
        return [int(key) for key, _ in self.coll.search(query, k, ef=self.ef)]

    def optimize(self, data_size: int | None = None):
        self._flush(0)
        self.db.optimize()
        self.db.wait_for_maintenance()

    def prepare_filter(self, filters):
        # Called in search processes before init(): open those read-only, so
        # any number of them can share the file.
        self.read_only = True
