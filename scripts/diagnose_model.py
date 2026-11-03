#!/usr/bin/env python3
"""Diagnostics for the cost model on results/measured/dataset.csv.

  python scripts/diagnose_model.py            # error tables + trivial-rule baselines
  python scripts/diagnose_model.py --noise    # also re-measures one config twice to estimate the measurement noise floor
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pidcomm_sim import cost_model as cm            # noqa: E402
from pidcomm_sim import experiments as ex           # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for f in ex.INT_FIELDS: r[f] = int(r[f])
        for f in ex.FLOAT_FIELDS: r[f] = float(r[f])
    return rows


def table(title, groups):
    print(f"\n{title}")
    for k in sorted(groups):
        e = np.array(groups[k]); print(f"  {str(k):<14} n={len(e):4d}  mean abs err {e.mean() * 100:5.1f}%  median {np.median(e) * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results" / "measured" / "dataset.csv"))
    ap.add_argument("--noise", action="store_true"); ap.add_argument("--cpu", type=int, default=None, help="pin to this logical CPU")
    ap.add_argument("--noise-reps", type=int, default=15); ap.add_argument("--simd", default="auto"); ap.add_argument("--stat", default="min", choices=["median", "min"]); a = ap.parse_args()
    ex.pin_cpu(a.cpu)
    if a.cpu is None and ex.performance_cores(): print(f"hint: hybrid CPU - performance cores are {ex.performance_cores()}; pin with --cpu N")
    rows = load(a.data)

    # leave-one-config-out predictions
    pred = {}
    for c in sorted({(r["shape"], r["mask"]) for r in rows}):
        m = cm.fit([r for r in rows if (r["shape"], r["mask"]) != c])
        for i, r in enumerate(rows):
            if (r["shape"], r["mask"]) == c: pred[i] = m.predict_ms(cm.row_counts(r))
    err = {i: abs(pred[i] - r["host_ms"]) / r["host_ms"] for i, r in enumerate(rows)}
    for title, keyf in (("error by primitive", lambda r: r["prim"]), ("error by level", lambda r: r["level"]),
                        ("error by bytes per PE", lambda r: r["bytes_per_pe"]), ("error by host time bucket (ms)",
                         lambda r: "<0.01" if r["host_ms"] < .01 else "<0.1" if r["host_ms"] < .1 else "<1" if r["host_ms"] < 1 else ">=1")):
        g = defaultdict(list)
        for i, r in enumerate(rows): g[keyf(r)].append(err[i])
        table(title, g)

    # trivial-rule baselines for the level-choice decision
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[cm._key(r)].append((i, r))
    n = hit_model = 0; hit_fixed = defaultdict(int)
    for g in groups.values():
        if len(g) < 2: continue
        meas = {r["level"]: r["host_ms"] for _, r in g}; mp = {r["level"]: pred[i] for i, r in g}
        best = min(meas, key=meas.get); n += 1; hit_model += min(mp, key=mp.get) == best
        for L in meas: hit_fixed[L] += best == L
    print(f"\nlevel choice over {n} decisions:  model {hit_model / n * 100:.0f}% correct")
    for L in sorted(hit_fixed): print(f"   rule 'always level {L}': {hit_fixed[L] / n * 100:.0f}% correct")

    worst = sorted(range(len(rows)), key=lambda i: -err[i])[:8]
    print("\n8 worst predictions (out-of-sample):")
    for i in worst:
        r = rows[i]; print(f"  {r['prim']:8s} L{r['level']} {r['shape']:>6s}[{r['mask']}] {r['bytes_per_pe']:>6d}B dt={r['dtype']} {r['op']:3s} "
                           f"measured {r['host_ms']:.4f} ms  predicted {pred[i]:.4f} ms")

    if a.noise:
        print(f"\nmeasurement noise floor: same configuration measured twice (reps={a.noise_reps}, stat={a.stat}, cpu pin={a.cpu})")
        kw = dict(shape=(8, 8), mask="11", sizes=(512, 2048, 8192, 32768), prims=("aa", "rs", "ar"), dtype=4, op="sum", reps=a.noise_reps, stat=a.stat, simd=a.simd)
        r1, r2 = ex.run_bench(**kw), ex.run_bench(**kw)
        d = np.array([abs(x["host_ms"] - y["host_ms"]) / min(x["host_ms"], y["host_ms"]) for x, y in zip(r1, r2)])
        print(f"   run-to-run difference: mean {d.mean() * 100:.1f}%  median {np.median(d) * 100:.1f}%  worst {d.max() * 100:.1f}%  ({len(d)} rows)")
        print("   -> if this is close to the model's error, the error is mostly measurement noise, not the model.")


if __name__ == "__main__":
    main()
