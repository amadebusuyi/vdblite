from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, MetricType


class VdbliteConfig(DBConfig):
    """Where the .vdb file lives; vdblite is embedded, so there is no server."""

    db_label: str = ""
    path: str

    def to_dict(self) -> dict:
        return {"path": self.path}


class VdbliteIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    m: int = 16
    ef_construction: int = 200
    # vdblite raises ef to at least k x 4 (400 for VectorDBBench's k=100).
    ef_search: int = 64
    quant: str = "int8"
    # Rows buffered per write transaction; VectorDBBench inserts 100 at a time.
    commit_every: int = 10_000

    def metric(self) -> str:
        return {MetricType.L2: "l2", MetricType.IP: "dot", MetricType.COSINE: "cosine"}.get(
            self.metric_type, "cosine"
        )

    def index_param(self) -> dict:
        return {
            "metric": self.metric(),
            "quant": self.quant,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }

    def search_param(self) -> dict:
        return {"ef": self.ef_search}
