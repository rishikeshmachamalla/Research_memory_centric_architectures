#!/usr/bin/env python3
"""Lines up OUR measured host-side results (results/*.csv from run_all.sh) against the numbers REPORTED in the PID-Comm paper.

The paper measured throughput on real UPMEM DIMMs (1024 PEs); we measure host-side time of a software model on one laptop
(<= 64 PEs, 'bus' = memcpy).  Absolute numbers are NOT comparable - only trends and ratios are.  Paper values are copied from
the text of Sec. VIII of the paper (Fig. numbers in brackets).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
PRIM = {"aa": "AlltoAll", "rs": "ReduceScatter", "ag": "AllGather", "ar": "AllReduce"}


def load(name):
    p = R / name
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("n_pe", "group", "bytes_per_pe", "dtype", "level"):
            if k in r: r[k] = int(r[k])
        for k in ("host_ms", "gbps", "comm_host_ms"):
            if k in r: r[k] = float(r[k])
    return rows


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def ms(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r["host_ms"]
    return None


def ratio(rows, a, b, **kw):          # time(level a) / time(level b)  == speedup of b over a
    x, y = ms(rows, level=a, **kw), ms(rows, level=b, **kw)
    return x / y if x and y else None


out, lines = {}, []
def row(metric, paper, ours, note=""):
    lines.append((metric, paper, ours, note))


s = load("sizes_1d64.csv")
if s:
    big = max(r["bytes_per_pe"] for r in s)
    sp = {p: ratio(s, 0, 3, prim=p, bytes_per_pe=big) for p in PRIM}
    out["speedup_full_over_base_64pe"] = sp
    row("Fig.14  AlltoAll speedup (full vs base)", "5.19x", f"{sp['aa']:.2f}x", f"64 PEs, {big} B/PE")
    row("Fig.14  ReduceScatter speedup", "4.46x", f"{sp['rs']:.2f}x")
    row("Fig.14  AllReduce speedup", "4.23x", f"{sp['ar']:.2f}x")
    row("Fig.14  AllGather speedup", "(small; baseline already fast)", f"{sp['ag']:.2f}x")
    row("Fig.14  geomean of these 4 primitives", "2.83x (all 8 primitives)", f"{gm(sp.values()):.2f}x", "not the same set of primitives")
    pr = gm([ratio(s, 0, 1, prim=p, bytes_per_pe=big) for p in PRIM])
    im = gm([ratio(s, 1, 2, prim=p, bytes_per_pe=big) for p in PRIM])
    cm_ = gm([ratio(s, 2, 3, prim=p, bytes_per_pe=big) for p in ("aa", "ag")])
    out["ablation_geomean"] = {"PR": pr, "IM": im, "CM(aa,ag)": cm_}
    row("Fig.16  +PE-assisted reordering (geomean)", "1.48x", f"{pr:.2f}x", "<1 means slower than base")
    row("Fig.16  +in-register modulation (additional)", "2.03x", f"{im:.2f}x")
    row("Fig.16  +cross-domain modulation (additional, AA/AG)", "1.42x", f"{cm_:.2f}x")
    sizes = sorted({r["bytes_per_pe"] for r in s})
    g = {b: gm([ratio(s, 0, 3, prim=p, bytes_per_pe=b) for p in PRIM]) for b in sizes}
    out["speedup_vs_size_geomean"] = g
    row("Fig.18  speedup grows with data size", "yes (2.89x geomean at 8 MB)", f"{g[sizes[0]]:.2f}x -> {g[sizes[-1]]:.2f}x", f"{sizes[0]} B -> {sizes[-1]} B per PE")

p = load("pe_sweep.csv")
if p:
    sz = max(r["bytes_per_pe"] for r in p); ns = sorted({r["n_pe"] for r in p})
    g = {n: gm([ratio(p, 0, 3, prim=q, n_pe=n, bytes_per_pe=sz) for q in PRIM]) for n in ns}
    out["speedup_vs_pes_geomean"] = g
    row("Fig.19  speedup grows with #PEs", "yes (64 -> 1024 PEs)", " -> ".join(f"{g[n]:.2f}x" for n in ns), f"{ns[0]} -> {ns[-1]} PEs, {sz} B/PE; we cannot reach 1024 PEs")

d = load("dtypes.csv")
if d:
    sp = {t: gm([ratio(d, 0, 3, prim=q, dtype=t) for q in ("rs", "ar")]) for t in (1, 4, 8)}
    out["rs_ar_speedup_by_dtype"] = sp
    row("Sec.VIII-F  8-bit data benefits more (RS/AR)", "yes (cross-domain modulation applies)", " / ".join(f"{t}B:{sp[t]:.2f}x" for t in (1, 4, 8)))

sh = load("shapes.csv")
if sh:
    keys = sorted({(r["shape"], r["mask"]) for r in sh})
    t = {q: [next(r["gbps"] for r in sh if r["prim"] == q and r["level"] == 3 and (r["shape"], r["mask"]) == k) for k in keys] for q in PRIM}
    out["throughput_by_shape_L3"] = {q: dict(zip(map(str, keys), v)) for q, v in t.items()}
    spread = {q: max(v) / min(v) for q, v in t.items()}
    row("Fig.20  AlltoAll/AllReduce throughput ~flat across shapes", "yes", f"max/min = {spread['aa']:.2f} (AA), {spread['ar']:.2f} (AR)", "closer to 1.00 = flatter")
    row("Fig.20  ReduceScatter/AllGather vary with shape", "yes (grow with x-axis length)", f"max/min = {spread['rs']:.2f} (RS), {spread['ag']:.2f} (AG)")

# ---- our additions ----
o = load("ops.csv")
if o:
    for q in ("rs", "ar"):
        s2, o2 = ms(o, prim=q, level=2, op="sum"), ms(o, prim=q, level=3, op="or")
        if s2 and o2: row(f"OURS  {PRIM[q]}: OR at level 3 vs sum at level 2", "n/a (paper: CM only non-arithmetic/8-bit)", f"{s2 / o2:.2f}x  (>1 = faster)", "no domain transfer needed for bitwise ops")
si = load("simd.csv")
if si:
    kinds = sorted({r["simd"] for r in si}); base = {k: gm([ms(si, prim=q, level=3, simd=k) for q in PRIM]) for k in kinds}
    row("OURS  host SIMD flavour (full design, geomean time)", "AVX-512 only", " | ".join(f"{k}: {base[k]:.3f} ms" for k in kinds), "lower is faster")
    out["simd_geomean_ms"] = base
    for isa in ("avx2", "avx512"):
        if isa in base and f"{isa}-ptr" in base:
            row(f"OURS  {isa}: loops compiled per ISA vs function pointers", "n/a", f"{base[isa + '-ptr'] / base[isa]:.2f}x  (>1 = faster)", "fix for AVX kernels not beating scalar")
    if "avx2" in base and "scalar" in base:
        row("OURS  avx2 (inlined) vs scalar", "n/a", f"{base['scalar'] / base['avx2']:.2f}x  (>1 = faster)")
b = load("bfs.csv")
if b:
    ns = sorted({r["n_pe"] for r in b})
    g = {n: next(r["comm_host_ms"] for r in b if r["n_pe"] == n and r["level"] == 0) / next(r["comm_host_ms"] for r in b if r["n_pe"] == n and r["level"] == 3) for n in ns}
    out["bfs_comm_speedup"] = g
    row("Fig.15  BFS speedup (we: communication only)", "1.20x-3.99x over all 5 apps, end-to-end", " -> ".join(f"{g[n]:.2f}x" for n in ns), f"{ns[0]} -> {ns[-1]} PEs; not end-to-end")

w = max(len(l[0]) for l in lines)
print(f"{'metric':<{w}} | {'paper (real HW)':<38} | ours (this laptop)")
print("-" * (w + 90))
for m, pa, ou, no in lines:
    print(f"{m:<{w}} | {pa:<38} | {ou}" + (f"   [{no}]" if no else ""))
json.dump(out, open(R / "paper_comparison.json", "w"), indent=2, default=str)
print("\nsaved", R / "paper_comparison.json")
