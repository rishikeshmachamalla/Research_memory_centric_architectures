# C++ backend

`src/` — the collective-communication engine (`collectives.hpp`), the simulated PIM memory model
(`pimsim.hpp`), host kernels in scalar/AVX2/AVX-512 (`kernels.hpp`), and five entry points:
`test` (correctness), `bench` (host-side timing sweep), `bfs`, `pcdump` (single-call, for the Python
cross-check), all built by `make`.

```bash
make            # builds test, bench, bfs, pcdump
./test          # ~18k correctness cases
./bench --shape 8x8 --mask 11 --sizes 4096,32768 --prims aa,rs,ag,ar --levels 0,1,2,3 --simd scalar
./bfs --n 64 --V 16384
```

`scripts/run_all.sh` sweeps everything needed for the paper-figure comparison; `scripts/compare_with_paper.py`
turns the resulting CSVs into the table printed in the report. `microbench/` holds the two small experiments
behind the SIMD-kernel section of the report (`run_microbench.sh`, `mutation_test.sh`).
