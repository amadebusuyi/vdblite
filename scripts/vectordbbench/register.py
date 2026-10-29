"""Registers the vdblite client in a VectorDBBench checkout (idempotent).

    python3 scripts/vectordbbench/register.py PATH/TO/VectorDBBench

VectorDBBench lists its databases in three dispatch methods of
`backend/clients/__init__.py` and in `cli/vectordbbench.py`; each vdblite
entry goes in next to Zvec's, the other embedded engine.
"""

import sys
from pathlib import Path

ZVEC = {
    "enum": '    Zvec = "Zvec"\n',
    "init": "        if self == DB.Zvec:\n            from .zvec.zvec import Zvec\n\n            return Zvec\n",
    "config": "        if self == DB.Zvec:\n            from .zvec.config import ZvecConfig\n\n            return ZvecConfig\n",
    "case": "        if self == DB.Zvec:\n            from .zvec.config import ZvecHNSWIndexConfig\n\n"
    "            return ZvecHNSWIndexConfig\n",
}
VDBLITE = {
    "enum": '    Vdblite = "vdblite"\n',
    "init": "        if self == DB.Vdblite:\n            from .vdblite.vdblite import Vdblite\n\n            return Vdblite\n\n",
    "config": "        if self == DB.Vdblite:\n            from .vdblite.config import VdbliteConfig\n\n"
    "            return VdbliteConfig\n\n",
    "case": "        if self == DB.Vdblite:\n            from .vdblite.config import VdbliteIndexConfig\n\n"
    "            return VdbliteIndexConfig\n\n",
}


def insert_before(text: str, anchor: str, new: str, path: Path) -> str:
    if new in text:
        return text
    if text.count(anchor) != 1:
        raise SystemExit(f"{path}: expected one {anchor.strip()[:50]!r}; VectorDBBench changed?")
    return text.replace(anchor, new + anchor)


def main() -> None:
    root = Path(sys.argv[1]) / "vectordb_bench"
    clients = root / "backend/clients/__init__.py"
    text = clients.read_text()
    for part in ("enum", "init", "config", "case"):
        text = insert_before(text, ZVEC[part], VDBLITE[part], clients)
    clients.write_text(text)

    cli = root / "cli/vectordbbench.py"
    text = cli.read_text()
    text = insert_before(
        text,
        "from ..backend.clients.zvec.cli import Zvec\n",
        "from ..backend.clients.vdblite.cli import Vdblite\n",
        cli,
    )
    text = insert_before(text, "cli.add_command(Zvec)\n", "cli.add_command(Vdblite)\n", cli)
    cli.write_text(text)
    print(f"registered vdblite in {root}")


if __name__ == "__main__":
    main()
