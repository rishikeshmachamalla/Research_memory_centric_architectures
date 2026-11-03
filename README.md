# PID-Comm in Software — Group 4 (CS5290 Assignment 2)

A software reproduction of the core of PID-Comm (ISCA'24): the virtual hypercube, four collectives
(AlltoAll, ReduceScatter, AllGather, AllReduce) at the paper's four implementation levels, a validated
analytical host-cost model, and a BFS application. Built because the authors' code crashed on the
UPMEM functional simulator on our machine (see `report/report.tex`, Sec. III-A, for details and the
exact error) — not because the task asked us to write our own simulator.

## Layout
- `src/pidcomm_sim/` — Python: topology, reference semantics, cost model, experiment drivers, BFS app.
- `cpp_backend/` — C++: the timed host-side engine (same semantics, independently implemented) + its own
  tests/benchmarks. See `cpp_backend/README.md`.
- `tests/` — pytest suite, including a word-for-word cross-check of the C++ backend against the Python reference.
- `scripts/` — experiment orchestration, the cost-model validator, and `check_report_claims.py`.
- `report/` — the IEEE-format LaTeX report and compiled PDF.

## Build & test everything
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib pytest
cd cpp_backend && make && cd ..
pytest -q                       # ~334 tests
cd cpp_backend && ./test        # ~18k correctness cases
```

## Reproduce all reported results (~25-30 min per run; find a performance core with `cat /sys/devices/cpu_core/cpus`)
```bash
SIMD=scalar CPU=<core> ./scripts/final_run.sh   # fair ablation vs. the paper (identical kernels at every level)
SIMD=avx2   CPU=<core> ./scripts/final_run.sh   # our AVX2 addition
python scripts/check_report_claims.py           # verifies report.tex's numbers against the measured results
```
Outputs land in `results/run_$SIMD/`, `results/final_$SIMD/` (text logs) and `cpp_backend/results/` (paper-figure sweep + `paper_comparison.json`/`.txt`).

## What's left outside the code
The report's contribution-statement section and the slides are not part of this archive.
