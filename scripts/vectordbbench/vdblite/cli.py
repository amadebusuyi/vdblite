from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class VdbliteTypedDict(CommonTypedDict):
    path: Annotated[str, click.option("--path", type=str, help="the .vdb file", required=True)]
    m: Annotated[int, click.option("--m", type=int, default=16, help="HNSW links per node")]
    ef_construction: Annotated[
        int, click.option("--ef-construction", type=int, default=200, help="HNSW build beam")
    ]
    ef_search: Annotated[
        int, click.option("--ef-search", type=int, default=64, help="HNSW search beam (raised to k x 4)")
    ]
    quant: Annotated[str, click.option("--quant", type=str, default="int8", help="int8, binary or none")]
    commit_every: Annotated[
        int, click.option("--commit-every", type=int, default=10_000, help="rows per write transaction")
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(VdbliteTypedDict)
def Vdblite(**parameters: Unpack[VdbliteTypedDict]):
    from .config import VdbliteConfig, VdbliteIndexConfig

    run(
        db=DB.Vdblite,
        db_config=VdbliteConfig(db_label=parameters["db_label"], path=parameters["path"]),
        db_case_config=VdbliteIndexConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            quant=parameters["quant"],
            commit_every=parameters["commit_every"],
        ),
        **parameters,
    )
