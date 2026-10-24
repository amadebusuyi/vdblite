"""Summarises run.sh results as Markdown: one table per dataset, with each
engine's fastest operating point at or above a recall target.

    python3 scripts/compare/embedded/report.py [DATASET ...]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "target/compare/embedded"
ENGINES = ["vecfile", "zvec", "lancedb-hnsw", "lancedb-ivfpq", "sqlite-vec"]
TARGETS = [0.95, 0.99]


def load(dataset, engine):
    d = OUT / dataset
    try:
        return json.loads((d / f"{engine}-build.json").read_text()), json.loads((d / f"{engine}-search.json").read_text())
    except FileNotFoundError:
        return None


def knob(k):
    return ", ".join(f"{a}={b}" for a, b in k.items()) or "exact"


def ms(us):
    return f"{us / 1000:.2f} ms" if us >= 1000 else f"{us:.0f} µs"


def at_recall(search, target):
    ok = [s for s in search["warm"] if s["recall"] >= target]
    if not ok:
        best = max(search["warm"], key=lambda s: s["recall"])
        return f"— (max {best['recall']:.3f})"
    s = min(ok, key=lambda s: s["p50_us"])
    return f"{ms(s['p50_us'])} / {ms(s['p99_us'])} ({knob(s['knob'])}, {s['recall']:.3f})"


def table(dataset):
    rows = []
    for e in ENGINES:
        r = load(dataset, e)
        if r is None:
            continue
        b, s = r
        disk = b.get("disk_compacted_mb", b["disk_mb"])
        disk_s = f"{disk:,.0f}" + (f" ({b['disk_mb']:,.0f} before compact)" if "disk_compacted_mb" in b else "")
        qps = s["qps"]
        ref = next((w for w in s["warm"] if w["knob"] == qps["knob"]), None)
        qps_s = f"{qps['qps']:,.0f}" + (f" (recall {ref['recall']:.3f})" if ref else "")
        rows.append([
            f"{e} {s['version']}",
            f"{b['ingest_s']:.0f} s",
            disk_s,
            f"{s['rss_mb']:,.0f}",
            f"{s['open_s'] * 1000:.0f} ms / {ms(s['cold']['p50_us'])}",
            *[at_recall(s, t) for t in TARGETS],
            qps_s,
            ms(s["floor_p50_us"]),
        ])
    head = ["Engine", "Load + index", "Disk MB", "Serving RSS MB", "Open / cold p50",
            *[f"p50 / p99 at recall ≥ {t}" for t in TARGETS], f"QPS, {s['qps']['threads']} threads", "Floor p50"]
    lines = [f"### {dataset}", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    sweeps = ["", "Full sweeps (recall@10, p50, p99):", ""]
    for e in ENGINES:
        r = load(dataset, e)
        if r:
            pts = "; ".join(f"{knob(w['knob'])}: {w['recall']:.4f}, {ms(w['p50_us'])}, {ms(w['p99_us'])}" for w in r[1]["warm"])
            sweeps.append(f"- {e}: {pts}")
    return "\n".join(lines + sweeps) + "\n"


if __name__ == "__main__":
    datasets = sys.argv[1:] or sorted(p.name for p in OUT.iterdir() if p.is_dir() and not p.name.endswith("-20000"))
    print("\n".join(table(d) for d in datasets))
