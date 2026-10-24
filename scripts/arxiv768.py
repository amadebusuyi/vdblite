"""Converts the arxiv768 parquet parts in DEST/raw to DEST/base.fvecs and
DEST/titles.txt. Run by scripts/fetch-arxiv768.sh."""

import struct
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DIM = 768
ROW = DIM * 4


def main() -> None:
    dest = Path(sys.argv[1])
    parts = int(sys.argv[2])
    header = struct.pack("<i", DIM)
    rows = 0
    with open(dest / "base.fvecs", "wb") as vecs, open(dest / "titles.txt", "w", encoding="utf-8") as titles:
        for i in range(parts):
            path = dest / "raw" / f"part-{i:02d}-arxiv-titles-instructorxl.parquet"
            table = pq.read_table(path, columns=["title", "vector"])
            lists = table.column("vector").combine_chunks()
            if lists.null_count or not pc.all(pc.equal(lists.value_lengths(), DIM)).as_py():
                raise SystemExit(f"{path}: every vector must have {DIM} dimensions")
            flat = lists.flatten()
            if flat.type != pa.float32() or flat.null_count:
                raise SystemExit(f"{path}: vectors must be non-null float32")
            # The raw little-endian float32 values, row after row.
            data = memoryview(flat.buffers()[1])[flat.offset * 4 : (flat.offset + len(flat)) * 4]
            for r in range(len(lists)):
                vecs.write(header)
                vecs.write(data[r * ROW : (r + 1) * ROW])
            for title in table.column("title").to_pylist():
                titles.write(" ".join((title or "").split()) + "\n")
            rows += len(lists)
            print(f"  {path.name}: {rows} rows so far", flush=True)
    print(f"wrote {rows} vectors")


if __name__ == "__main__":
    main()
