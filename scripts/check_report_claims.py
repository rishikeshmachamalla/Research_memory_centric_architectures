#!/usr/bin/env python3
"""Sanity check: extract every number-looking claim this report script PRINTS and compare it against the
numbers already sitting in report.tex, so a hand-copy mistake shows up before submission.

This does NOT re-measure anything (it doesn't touch the laptop's timing); it recomputes ratios straight from
results/run_scalar, results/run_avx2 and results/bfs and diffs them against the \newcommand values / table
numbers typed into report/report.tex. Run AFTER both final_run.sh runs and AFTER you've hand-edited the report.

    python scripts/check_report_claims.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.tex"

def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def pct(x): return f"{x*100:.1f}%"

def check(label, computed, pattern, tolerance=0.06):
    """pattern: a regex that must appear in report.tex; its first captured number is compared to `computed`
    (expressed as the same units, e.g. 14.5 for 14.5%, or 1.38 for 1.38x)."""
    text = REPORT.read_text()
    m = re.search(pattern, text)
    status = "MISSING"
    if m:
        try:
            reported = float(m.group(1))
            rel = abs(reported - computed) / max(abs(computed), 1e-9)
            status = "OK" if rel <= tolerance else f"MISMATCH (report says {reported}, computed {computed:.2f})"
        except Exception:
            status = f"found text but couldn't parse number near: {m.group(0)!r}"
    print(f"  [{status:<45}] {label}")
    return status == "OK"

def main():
    ok = True
    scalar = load_json(ROOT / "results" / "run_scalar" / "model" / "validation.json")
    avx2   = load_json(ROOT / "results" / "run_avx2"   / "model" / "validation.json")
    bfs    = load_json(ROOT / "results" / "run_scalar" / "bfs"   / "bfs_summary.json")
    # NOTE: cpp_backend/results/ is overwritten by whichever final_run.sh (scalar/avx2) ran LAST, so Table I
    # (quoted from the SCALAR run) must be read from the archived per-run copy, not the live shared folder.
    cmp_path = ROOT / "results" / "final_scalar" / "cpp_results" / "paper_comparison.json"
    if not cmp_path.exists():
        cmp_path = ROOT / "cpp_backend" / "results" / "paper_comparison.json"  # fallback if archive is missing
        print(f"  (warning: {ROOT / 'results' / 'final_scalar' / 'cpp_results' / 'paper_comparison.json'} not found;"
              f" falling back to the LIVE cpp_backend/results/, which may belong to whichever SIMD run was done last)")
    cmp_j  = load_json(cmp_path)

    if not (scalar and avx2):
        print("Run both  SIMD=scalar/avx2 CPU=N ./scripts/final_run.sh  first — no results/run_* found.")
        return 2

    print("Cost-model numbers (Table II):")
    ok &= check("scalar leave-one-config-out MAPE", scalar["leave_one_config_out"]["mape"] * 100,
                r"Leave-one-config-out & ([\d.]+)\\%")
    ok &= check("avx2 leave-one-config-out MAPE", avx2["leave_one_config_out"]["mape"] * 100,
                r"Leave-one-config-out & [\d.]+\\% & ([\d.]+)\\%")
    ok &= check("scalar level-choice accuracy", scalar["leave_one_config_out_level_choice"]["exact_match"] * 100,
                r"Level choice correct & ([\d.]+)\\%")
    ok &= check("avx2 level-choice accuracy", avx2["leave_one_config_out_level_choice"]["exact_match"] * 100,
                r"Level choice correct & [\d.]+\\% & ([\d.]+)\\%")

    if cmp_j:
        print(f"\nPaper-comparison numbers (Table I), from {cmp_path.relative_to(ROOT)}:")
        sp = cmp_j.get("speedup_full_over_base_64pe", {})
        for prim, texname in (("aa", "AlltoAll"), ("rs", "ReduceScatter"), ("ar", "AllReduce")):  # AllGather has no paper value in the report to check against
            if prim in sp and sp[prim]:
                ok &= check(f"{texname} speedup", sp[prim], rf"{texname} speedup & [\d.]+\$\\times\$ & ([\d.]+)\$\\times\$")

    bfs_csv = ROOT / "results" / "run_scalar" / "bfs" / "bfs_results.csv"
    if bfs_csv.exists():
        import csv
        rows = list(csv.DictReader(open(bfs_csv)))
        errs = [abs(float(r["predicted_ms"]) - float(r["comm_host_ms"])) / float(r["comm_host_ms"])
                for r in rows if r.get("predicted_ms")]
        if errs:
            print("\nBFS numbers (from results/run_scalar/bfs/bfs_results.csv):")
            mean_err = sum(errs) / len(errs) * 100
            ok &= check("BFS model error (mean, scalar)", mean_err, r"BFS \(unseen application\) & ([\d.]+)\\%")

    print("\n" + ("ALL CHECKED NUMBERS MATCH (within 6%)." if ok else
                  "SOME NUMBERS DID NOT MATCH OR WERE NOT FOUND -- read the lines above and fix report.tex before submitting."))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
