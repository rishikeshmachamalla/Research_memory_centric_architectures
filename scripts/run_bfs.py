#!/usr/bin/env python3
"""BFS application experiment.

  python scripts/run_bfs.py --stat min --runs 15 --cpu 2

For n_pe = 8..64 it runs the C++ BFS at levels 0..3 (MEASURED host-side communication time, summed over all BFS rounds)
and compares it with the cost model's prediction (MODELED).  The model is fitted on results/measured/dataset.csv, which
contains NO BFS runs, so this is a validation on an application the model has never seen.
Only AllReduce communication is timed: PE-side kernel time is not available without hardware, so no end-to-end speedup is claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pidcomm_sim import bfs_app as app              # noqa: E402
from pidcomm_sim import cost_model as cm            # noqa: E402
from pidcomm_sim import experiments as ex           # noqa: E402

LV = {0: "Base", 1: "+PR", 2: "+PR+IM", 3: "+PR+IM+CM"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=16384); ap.add_argument("--deg", type=int, default=8)
    ap.add_argument("--runs", type=int, default=15); ap.add_argument("--stat", default="min", choices=["median", "min"])
    ap.add_argument("--cpu", type=int, default=None); ap.add_argument("--simd", default="auto")
    ap.add_argument("--pes", default="8,16,32,64"); ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--data", default=str(ROOT / "results" / "measured" / "dataset.csv"))
    a = ap.parse_args()
    ex.pin_cpu(a.cpu)
    out = Path(a.out) / "bfs"; out.mkdir(parents=True, exist_ok=True)

    model = None
    if Path(a.data).exists():
        rows = list(csv.DictReader(open(a.data)))
        for r in rows:
            for f in ex.INT_FIELDS: r[f] = int(r[f])
            for f in ex.FLOAT_FIELDS: r[f] = float(r[f])
        model = cm.fit(rows)
    else:
        print("note: no dataset.csv - model predictions skipped (run scripts/validate_model.py first)")

    # one fixed graph for all PE counts, and an independent Python check of the result
    edges = app.random_graph(a.V, a.deg, seed=2024); g = out / "graph.txt"; app.write_graph(g, a.V, edges)
    reached, dsum = app.summary(app.reference_bfs(a.V, edges))
    allrows = []
    for n in [int(x) for x in a.pes.split(",")]:
        for r in ex.run_bfs(n, a.V, a.deg, a.runs, a.stat, a.simd, graph=g):
            assert r["correct"] == 1 and (r["reached"], r["dist_sum"]) == (reached, dsum), "BFS result mismatch!"
            if model:
                c = cm.counts("ar", r["level"], n, n, (a.V // 64) * 8, 8, "or")
                r["predicted_ms"] = r["iters"] * model.predict_ms(c)
            allrows.append(r)
    keys = ["n_pe", "V", "edges", "level", "iters", "comm_host_ms", "predicted_ms", "bus_rd", "bus_wr", "dt", "correct", "reached", "dist_sum"]
    with open(out / "bfs_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(allrows)

    print(f"BFS: V={a.V}, E={len(edges)}, reached {reached} vertices; all runs match the independent Python BFS\n")
    print(f"{'PEs':>4} {'level':<10} {'rounds':>6} {'measured ms':>12} {'predicted ms':>13} {'err':>6} {'DTs':>7}")
    for r in allrows:
        pm = r.get("predicted_ms"); err = f"{abs(pm - r['comm_host_ms']) / r['comm_host_ms'] * 100:5.0f}%" if pm else "   n/a"
        print(f"{r['n_pe']:>4} {LV[r['level']]:<10} {r['iters']:>6} {r['comm_host_ms']:>12.3f} {(pm if pm else float('nan')):>13.3f} {err:>6} {r['dt']:>7}")
    print("\ncommunication speedup of the full design over the baseline (level 3 vs level 0):")
    summ = {}
    for n in sorted({r["n_pe"] for r in allrows}):
        d = {r["level"]: r for r in allrows if r["n_pe"] == n}
        m = d[0]["comm_host_ms"] / d[3]["comm_host_ms"]; p = d[0].get("predicted_ms", 0) / d[3]["predicted_ms"] if model else float("nan")
        summ[n] = {"measured_speedup": m, "predicted_speedup": p}
        print(f"   {n:>3} PEs: measured {m:5.2f}x   predicted {p:5.2f}x")
    if model:
        e = [abs(r["predicted_ms"] - r["comm_host_ms"]) / r["comm_host_ms"] for r in allrows]
        print(f"\nmodel error on BFS (never in training): mean {np.mean(e) * 100:.1f}%  median {np.median(e) * 100:.1f}%  max {np.max(e) * 100:.1f}%")
    json.dump({"args": vars(a), "speedups": summ, "env": ex.environment()}, open(out / "bfs_summary.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.6, 3.6)); ns = sorted({r["n_pe"] for r in allrows})
        for L in range(4):
            ax.plot(ns, [next(r["comm_host_ms"] for r in allrows if r["n_pe"] == n and r["level"] == L) for n in ns], marker="o", label=LV[L])
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("#PEs"); ax.set_ylabel("host-side communication time (ms)")
        ax.set_title("BFS, AllReduce(OR) - measured"); ax.legend(); fig.tight_layout(); fig.savefig(out / "fig_bfs.png", dpi=150)
    except ImportError:
        pass
    print("written to", out)


if __name__ == "__main__":
    main()
