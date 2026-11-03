#!/usr/bin/env python3
"""Measure host runtimes with the C++ backend, fit the analytical cost model, validate it out-of-sample.

  python scripts/validate_model.py                 # full run (a few minutes)
  python scripts/validate_model.py --quick         # smaller sweep
  python scripts/validate_model.py --reuse         # reuse results/measured/dataset.csv

Outputs (results/):  measured/dataset.csv, measured/environment.json, model/validation.json,
                     model/speedups.csv, model/fig_pred_vs_measured.png, model/fig_speedup.png
Categories are kept apart: MEASURED = laptop host runtime of the C++ kernels; MODELED = cost-model prediction;
neither is UPMEM hardware time.
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
from pidcomm_sim import cost_model as cm            # noqa: E402
from pidcomm_sim import experiments as ex           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--simd", default="auto")
    ap.add_argument("--stat", default="median", choices=["median", "min"]); ap.add_argument("--cpu", type=int, default=None, help="pin to this logical CPU")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    ex.pin_cpu(a.cpu)
    if a.cpu is None and ex.performance_cores(): print(f"hint: hybrid CPU - performance cores are {ex.performance_cores()}; pin with --cpu N")
    out = Path(a.out); (out / "measured").mkdir(parents=True, exist_ok=True); (out / "model").mkdir(parents=True, exist_ok=True)
    ds = out / "measured" / "dataset.csv"

    if a.reuse and ds.exists():
        rows = list(csv.DictReader(open(ds)))
        for r in rows:
            for f in ex.INT_FIELDS: r[f] = int(r[f])
            for f in ex.FLOAT_FIELDS: r[f] = float(r[f])
    else:
        print("measuring host runtimes (C++ backend)...")
        rows = ex.build_dataset(a.quick, a.reps, a.simd, stat=a.stat)
        with open(ds, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        env = ex.environment(); env.update(simd=rows[0]["simd"], reps=a.reps, stat=a.stat, cpu_pin=a.cpu, quick=a.quick, rows=len(rows))
        json.dump(env, open(out / "measured" / "environment.json", "w"), indent=2)
    print(f"{len(rows)} measurements")

    # 1. event counts vs C++ counters (exact)
    bad = [r for r in rows if {k: r[k] for k in ("bus_rd", "bus_wr", "dt", "host_mem_blk")} !=
           {"bus_rd": (c := cm.row_counts(r))["rd"], "bus_wr": c["wr"], "dt": c["dt"], "host_mem_blk": c["mem"]}]
    print(f"[counts] formulas == C++ counters on {len(rows) - len(bad)}/{len(rows)} rows")

    # 2. time model: in-sample, leave-one-configuration-out, size extrapolation
    full = cm.fit(rows)
    res = {"counts_exact_rows": len(rows) - len(bad), "rows": len(rows), "coefficients_ms_per_event": full.as_dict(),
           "in_sample": cm.evaluate(full, rows), "in_sample_level_choice": cm.level_choice_accuracy(full, rows)}
    cfgs = sorted({(r["shape"], r["mask"]) for r in rows})
    cv_pred, cv_rows, cv_models = [], [], []
    for c in cfgs:                                     # held-out = an entire hypercube shape/mask the fit never saw
        tr = [r for r in rows if (r["shape"], r["mask"]) != c]; te = [r for r in rows if (r["shape"], r["mask"]) == c]
        if not tr or not te: continue
        m = cm.fit(tr); cv_models.append((m, te)); cv_rows += te; cv_pred += [m.predict_ms(cm.row_counts(r)) for r in te]
    y = np.array([r["host_ms"] for r in cv_rows]); p = np.array(cv_pred); rel = np.abs(p - y) / y
    res["leave_one_config_out"] = {"n": len(y), "mape": float(rel.mean()), "median_ape": float(np.median(rel)), "p90_ape": float(np.quantile(rel, .9))}
    hits = regs = n = 0; reg = []
    for m, te in cv_models:
        lc = cm.level_choice_accuracy(m, te)
        if lc["decisions"]: n += lc["decisions"]; hits += round(lc["exact_match"] * lc["decisions"]); reg.append(lc["max_regret"])
    res["leave_one_config_out_level_choice"] = {"decisions": n, "exact_match": hits / n if n else None, "max_regret": max(reg) if reg else None}
    cut = sorted({r["bytes_per_pe"] for r in rows})[-2]
    tr, te = [r for r in rows if r["bytes_per_pe"] <= cut // 4], [r for r in rows if r["bytes_per_pe"] >= cut]
    if tr and te:
        res["size_extrapolation"] = {"train_max_bytes": cut // 4, "test_min_bytes": cut, **cm.evaluate(cm.fit(tr), te)}
    sp = [s for m, te in cv_models for s in cm.speedup_errors(m, te)]
    res["speedup_L3_over_L0_leave_one_config_out"] = {"n": len(sp), "mean_rel_err": float(np.mean([s["rel_err"] for s in sp])) if sp else None}
    json.dump(res, open(out / "model" / "validation.json", "w"), indent=2)
    if sp:
        with open(out / "model" / "speedups.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sp[0].keys())); w.writeheader(); w.writerows(sp)
    print(json.dumps(res, indent=2))

    # 3. figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.4)); lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.loglog([lo, hi], [lo, hi], "k--", lw=.8)
    for pr in cm.PRIMS:
        idx = [i for i, r in enumerate(cv_rows) if r["prim"] == pr]
        if idx: ax.scatter(y[idx], p[idx], s=9, label=pr)
    ax.set_xlabel("measured host time (ms)"); ax.set_ylabel("model prediction (ms)")
    ax.set_title(f"leave-one-config-out, MAPE {rel.mean() * 100:.0f}%"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "model" / "fig_pred_vs_measured.png", dpi=150)
    if sp:
        fig, ax = plt.subplots(figsize=(4.8, 4.4)); ms, ps = [s["measured"] for s in sp], [s["predicted"] for s in sp]
        lim = max(max(ms), max(ps)) * 1.1; ax.plot([0, lim], [0, lim], "k--", lw=.8)
        for pr in cm.PRIMS:
            pts = [(s["measured"], s["predicted"]) for s in sp if s["prim"] == pr]
            if pts: ax.scatter(*zip(*pts), s=12, label=pr)
        ax.set_xlabel("measured speedup (Base / full PID-Comm)"); ax.set_ylabel("predicted speedup"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(out / "model" / "fig_speedup.png", dpi=150)
    print("results written to", out)


if __name__ == "__main__":
    main()
