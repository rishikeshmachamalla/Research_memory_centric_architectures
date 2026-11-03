#!/usr/bin/env python3
"""Does a richer cost model help?  Tests extra features with NESTED selection (honest held-out error).

  python scripts/explore_features.py            # uses results/measured/dataset.csv

Candidates added to the 7 base features (bus dt mem rot perm comb swords):
  call        : constant 1 per primitive call (fixed overhead: allocation, loop setup)
  spillA(T)   : mem-blocks that occur when the staging working set (2 * n_pe * bytes_per_pe) exceeds T bytes
  spillB(T)   : mem + swords + bus blocks that occur above T   (everything that streams through memory)
For every held-out hypercube config the candidate is chosen using ONLY the other configs (inner leave-one-config-out),
so the reported error is not tuned on the data it is scored on.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pidcomm_sim import cost_model as cm            # noqa: E402
from pidcomm_sim import experiments as ex           # noqa: E402

THRESHOLDS = [2 ** k for k in range(17, 23)]        # 128 KiB .. 4 MiB


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for f in ex.INT_FIELDS: r[f] = int(r[f])
        for f in ex.FLOAT_FIELDS: r[f] = float(r[f])
    return rows


def candidates():
    cands = {"base": lambda c, r: []}
    cands["+call"] = lambda c, r: [1.0]
    for T in THRESHOLDS:
        ws = lambda r: 2 * r["n_pe"] * r["bytes_per_pe"]
        cands[f"+spillA({T >> 10}K)"] = lambda c, r, T=T, ws=ws: [c["mem"] if ws(r) > T else 0.0]
        cands[f"+spillB({T >> 10}K)"] = lambda c, r, T=T, ws=ws: [(c["mem"] + c["swords"] + c["bus"]) if ws(r) > T else 0.0]
        cands[f"+call+spillB({T >> 10}K)"] = lambda c, r, T=T, ws=ws: [1.0, (c["mem"] + c["swords"] + c["bus"]) if ws(r) > T else 0.0]
    return cands


def design(rows, fn):
    out = []
    for r in rows:
        c = cm.row_counts(r)
        out.append(np.concatenate([cm.feature_vector(c), fn(c, r)]))
    return np.array(out)


def fit_predict(Atr, ytr, Ate):
    coef, _ = nnls(Atr / ytr[:, None], np.ones(len(ytr)))
    return Ate @ coef, coef


def loco(X, y, cfg):
    errs = np.zeros(len(y))
    for c in sorted(set(cfg)):
        te = np.array([k == c for k in cfg]); p, _ = fit_predict(X[~te], y[~te], X[te]); errs[te] = np.abs(p - y[te]) / y[te]
    return errs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(ROOT / "results" / "measured" / "dataset.csv")); a = ap.parse_args()
    rows = load(a.data); y = np.array([r["host_ms"] for r in rows]); cfg = [(r["shape"], r["mask"]) for r in rows]
    cands = candidates(); X = {k: design(rows, fn) for k, fn in cands.items()}

    print(f"{len(rows)} rows, {len(set(cfg))} hypercube configs")
    print("\nplain leave-one-config-out error of every candidate (optimistic: chosen on the same data):")
    plain = {k: loco(X[k], y, cfg) for k in X}
    for k in sorted(plain, key=lambda k: plain[k].mean())[:8]: print(f"   {k:<22} mean {plain[k].mean() * 100:5.1f}%  median {np.median(plain[k]) * 100:5.1f}%")
    print(f"   {'base':<22} mean {plain['base'].mean() * 100:5.1f}%  median {np.median(plain['base']) * 100:5.1f}%   <- current model")

    # nested selection: for each outer config pick the candidate with the best INNER loco error on the training configs
    outer = np.zeros(len(y)); chosen = defaultdict(int)
    for c in sorted(set(cfg)):
        te = np.array([k == c for k in cfg]); tr = ~te; cfg_tr = [k for k, t in zip(cfg, tr) if t]
        best = min(X, key=lambda k: loco(X[k][tr], y[tr], cfg_tr).mean()); chosen[best] += 1
        p, _ = fit_predict(X[best][tr], y[tr], X[best][te]); outer[te] = np.abs(p - y[te]) / y[te]
    print("\nNESTED (honest) held-out error:")
    print(f"   with feature selection   mean {outer.mean() * 100:5.1f}%  median {np.median(outer) * 100:5.1f}%  p90 {np.quantile(outer, .9) * 100:5.1f}%")
    print(f"   base model only          mean {plain['base'].mean() * 100:5.1f}%  median {np.median(plain['base']) * 100:5.1f}%  p90 {np.quantile(plain['base'], .9) * 100:5.1f}%")
    print("   candidates chosen per held-out config:", dict(chosen))

    # trivial rules vs measured best level, and error by size
    groups = defaultdict(list)
    for r in rows: groups[cm._key(r)].append(r)
    dec = [g for g in groups.values() if len(g) > 1]; fixed = defaultdict(int)
    for g in dec:
        best = min(g, key=lambda r: r["host_ms"])["level"]
        for L in {r["level"] for r in g}: fixed[L] += best == L
    print(f"\nbaseline rules over {len(dec)} level decisions: " + ", ".join(f"always L{L}: {fixed[L] / len(dec) * 100:.0f}%" for L in sorted(fixed)))
    bysz = defaultdict(list)
    for r, e in zip(rows, plain["base"]): bysz[r["bytes_per_pe"]].append(e)
    print("base-model error by bytes per PE: " + ", ".join(f"{k}B {np.mean(v) * 100:.0f}%" for k, v in sorted(bysz.items())))


if __name__ == "__main__":
    main()
