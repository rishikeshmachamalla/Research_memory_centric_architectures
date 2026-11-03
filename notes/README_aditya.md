# Baseline Results doc — for Aditya

`baseline/baseline_results.pdf` (and its `.tex` source) is the standalone
"Baseline Results: graphs and/or tables showing the baseline results" item
from the MS Form. Separate from the main report and from Roopesh's
Contribution Results doc (which covers the cost model / bitwise-reduction /
AVX2 findings — the "add something new" part).

## What's already in it, and verified
- Correctness summary table (17,976 C++ cases, 334 Python tests, the
  fault-injection proof the cross-check isn't vacuous)
- **Table I** — full paper-vs-ours comparison (all 4 primitives, all 3
  optimization steps) — exact numbers, same ones already checked against
  your measured results by `scripts/check_report_claims.py`
- Data-size sensitivity, PE-count sensitivity, element-width sensitivity,
  hypercube-shape sensitivity — tables with real numbers
- **Full BFS baseline table** (8/16/32/64 PEs × 4 levels, all real data)

Compiled clean: 0 errors, 0 warnings, verified with a full `latexmk -C`
+ fresh rebuild (not relying on a stale cached pass), and every page visually
inspected at full resolution. One real alignment bug was caught and fixed
during that check (a footnote was moved out of a data column) — worth
knowing in case you edit numbers yourself later: **always rebuild AND
re-open the PDF after editing**, don't just trust that `latexmk` "succeeded."

## What YOU need to add — 5 figures, all already exist on your/Rishikesh's machine
These came from `cpp_backend/scripts/plot.py` after `scripts/run_all.sh` —
I was never sent the raw files, only summary numbers, so I built honest
placeholders instead of guessing. Copy these into `baseline/figures/`
(exact filenames, from wherever `run_all.sh` wrote them — likely
`cpp_backend/results/` and `results/bfs/`):

| Copy this file | To |
|---|---|
| `fig_ablation.png` | `baseline/figures/fig_ablation.png` |
| `fig_datasize.png` | `baseline/figures/fig_datasize.png` |
| `fig_pe_sweep.png` | `baseline/figures/fig_pe_sweep.png` |
| `fig_shapes.png` | `baseline/figures/fig_shapes.png` |
| `fig_bfs.png` | `baseline/figures/fig_bfs.png` |

If any of these don't exist yet, re-run (from the project root, venv active):
```
cd cpp_backend && SIMD=scalar CPU=<your perf core> ./scripts/run_all.sh
python scripts/plot.py
cd .. && python scripts/run_bfs.py --stat min --runs 15 --cpu <core>
```

**Why two of the tables look sparse (data-size, PE-count):** I only had the
smallest/largest endpoint values from the summary comparison, not the full
sweep — I will NOT fabricate the missing middle data points. The real
5-point and 4-point curves exist in `fig_datasize.png` / `fig_pe_sweep.png`
once you copy them in; they'll show the full picture my 2-point tables
can't.

## Recompile after adding figures
```
cd baseline && latexmk -C && latexmk -pdf baseline_results.tex
```
Then **open the PDF and look at it** — don't just check the exit code.

## Your second job: proofreading pass on the main report
Once this doc and the figures are in, give `report/report.tex` /
`report/report.pdf` (from Rishikesh) a careful read for typos, awkward
phrasing, and number-sanity (cross-reference anything that looks off
against `scripts/check_report_claims.py`'s output rather than trusting it
on sight).
