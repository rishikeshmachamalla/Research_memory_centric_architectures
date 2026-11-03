# Contribution Results doc — for Roopesh (your second deliverable)

`contribution/contribution_results.pdf` (+ `.tex` source) is the standalone
"Contribution Results: graphs and/or tables demonstrating the proposed
experimental results" item from the MS Form. Covers only the three NEW
additions (cost model, bitwise-reduction finding, AVX2 study) — baseline
reproduction against the paper is Aditya's separate Baseline Results doc.

## What's in it, and verified
- **Cost model**: exact event-count match (856/856, both SIMD builds),
  out-of-sample error table (held-out shapes, size extrapolation, BFS),
  and the honest negative result (doesn't beat "always pick L3")
- **Bitwise reduction**: OR/XOR skip-DT speedups, scalar AND avx2 builds
  shown side by side (a genuine extra data point the main report doesn't
  show — both are real measurements, not invented)
- **AVX2 study**: both microbenchmarks in full, the refuted hypothesis,
  the real cause, the fix's effect, and the honest "doesn't reliably beat
  scalar" conclusion (sign flip between two measurement sessions)

Compiled clean: single-pass, 0 errors/warnings, confirmed via fresh
`latexmk -C` rebuild (not a stale cached pass), every page visually
inspected at full resolution — including the two side-by-side microbenchmark
tables on page 3, the trickiest layout, which renders cleanly.

## What YOU need to add — 2 figures
| Copy this file | To | Source |
|---|---|---|
| `fig_pred_vs_measured.png` | `contribution/figures/` | `results/run_scalar/model/` (also at `results/run_avx2/model/` if you want both) |
| `fig_simd.png` | `contribution/figures/` | `cpp_backend/results/` (from `scripts/plot.py`) |

If missing, regenerate (project root, venv active):
```
python scripts/validate_model.py --quick --reps 15 --stat min --cpu <core> --simd scalar
cd cpp_backend && ./scripts/run_all.sh && python scripts/plot.py && cd ..
```

## Recompile after adding figures
```
cd contribution && latexmk -C && latexmk -pdf contribution_results.tex
```
Then **open the PDF and actually look at it** — same discipline as the
Baseline doc: don't trust a clean exit code alone.

## One thing worth knowing before you present
The AVX2-build bitwise-reduction numbers (3.24x/3.82x) are noticeably
higher than the scalar ones (1.60x/1.89x) quoted as our headline claim.
Both are real, separately-measured numbers — not an error — but if asked,
the honest answer is: the scalar numbers isolate the algorithmic effect
(skipping DT) cleanly, while the AVX2 numbers also include whatever the
AVX2 kernels themselves contribute, which Section 3 of this same doc shows
is NOT a reliable, one-directional effect on our machine.
