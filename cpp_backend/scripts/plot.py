#!/usr/bin/env python3
"""Turns results/*.csv into figures in results/fig_*.png  (matplotlib only)."""
import csv, os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
def load(name):
    p = os.path.join(R, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
LV = {0: "Base", 1: "+PR", 2: "+PR+IM", 3: "+PR+IM+CM"}
PR = ["aa", "rs", "ag", "ar"]; PN = {"aa": "AlltoAll", "rs": "ReduceScatter", "ag": "AllGather", "ar": "AllReduce", "arnaive": "AllReduce (RS+AG)"}

def f(r, k): return float(r[k])

# Fig A: ablation (analogue of paper Fig. 16) at the largest size
rows = load("sizes_1d64.csv")
if rows:
    big = max(int(r["bytes_per_pe"]) for r in rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for li, L in enumerate([0, 1, 2, 3]):
        ys = [next((f(r, "gbps") for r in rows if r["prim"] == p and int(r["level"]) == L and int(r["bytes_per_pe"]) == big), 0) for p in PR]
        ax.bar([i + li * 0.2 for i in range(4)], ys, 0.2, label=LV[L])
    ax.set_xticks([i + 0.3 for i in range(4)]); ax.set_xticklabels([PN[p] for p in PR]); ax.set_ylabel("host-side throughput (GB/s)")
    ax.set_title(f"Ablation, 64 PEs, {big} B/PE"); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_ablation.png"), dpi=150)
    # Fig B: throughput vs size (analogue of Fig. 18)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, p in zip(axs, PR):
        for L in (0, 3):
            pts = sorted((int(r["bytes_per_pe"]), f(r, "gbps")) for r in rows if r["prim"] == p and int(r["level"]) == L)
            ax.plot(*zip(*pts), marker="o", label=LV[L])
        ax.set_xscale("log", base=2); ax.set_title(PN[p]); ax.set_xlabel("bytes per PE")
    axs[0].set_ylabel("GB/s"); axs[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_datasize.png"), dpi=150)

# Fig C: speedup of full PID-Comm over baseline vs #PEs (analogue of Fig. 19)
rows = load("pe_sweep.csv")
if rows:
    fig, ax = plt.subplots(figsize=(6, 3.4)); size = max(int(r["bytes_per_pe"]) for r in rows)
    for p in PR:
        xs = sorted({int(r["n_pe"]) for r in rows}); ys = []
        for n in xs:
            g = {int(r["level"]): f(r, "host_ms") for r in rows if r["prim"] == p and int(r["n_pe"]) == n and int(r["bytes_per_pe"]) == size}
            ys.append(g[0] / g[3] if 0 in g and 3 in g else 0)
        ax.plot(xs, ys, marker="o", label=PN[p])
    ax.set_xlabel("#PEs"); ax.set_ylabel("speedup of host-side time"); ax.set_xscale("log", base=2); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_pe_sweep.png"), dpi=150)

# Fig D: shapes
rows = load("shapes.csv")
if rows:
    keys = sorted({(r["shape"], r["mask"]) for r in rows}); fig, ax = plt.subplots(figsize=(9, 3.4))
    for pi, p in enumerate(PR):
        ys = []
        for k in keys:
            g = {int(r["level"]): f(r, "host_ms") for r in rows if r["prim"] == p and (r["shape"], r["mask"]) == k}
            ys.append(g[0] / g[3] if 0 in g and 3 in g else 0)
        ax.bar([i + pi * 0.2 for i in range(len(keys))], ys, 0.2, label=PN[p])
    ax.set_xticks([i + 0.3 for i in range(len(keys))]); ax.set_xticklabels([f"{a}\n[{b}]" for a, b in keys], fontsize=7)
    ax.set_ylabel("speedup (Base / full)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_shapes.png"), dpi=150)

# Fig E: SIMD flavour (our addition)
rows = load("simd.csv")
if rows:
    sm = sorted({r["simd"] for r in rows}); fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for si, s in enumerate(sm):
        ys = [next((f(r, "gbps") for r in rows if r["simd"] == s and r["prim"] == p and int(r["level"]) == 3), 0) for p in PR]
        ax.bar([i + si * 0.25 for i in range(4)], ys, 0.25, label=s)
    ax.set_xticks([i + 0.25 for i in range(4)]); ax.set_xticklabels([PN[p] for p in PR], fontsize=8); ax.set_ylabel("GB/s (full PID-Comm)"); ax.legend(title="host SIMD")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_simd.png"), dpi=150)

# Fig F: BFS
rows = load("bfs.csv")
if rows:
    ns = sorted({int(r["n_pe"]) for r in rows}); fig, ax = plt.subplots(figsize=(6, 3.4))
    for L in range(4):
        ax.plot(ns, [next(f(r, "comm_host_ms") for r in rows if int(r["n_pe"]) == n and int(r["level"]) == L) for n in ns], marker="o", label=LV[L])
    ax.set_xscale("log", base=2); ax.set_xlabel("#PEs"); ax.set_ylabel("host-side communication time (ms)"); ax.set_title("BFS, AllReduce(OR)"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_bfs.png"), dpi=150)
print("figures written to", os.path.abspath(R))
