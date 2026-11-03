#!/usr/bin/env bash
# Runs the whole evaluation. Output: results/*.csv   (takes a few minutes)
set -e
cd "$(dirname "$0")/.."
make -s all
mkdir -p results; rm -f results/*.csv
SIMD=${SIMD:-scalar}          # scalar = identical kernels at every level, so the ablation isolates the PID-Comm technique (fair vs the paper)
REPS=${REPS:-15}
STAT=${STAT:-min}             # min of the repetitions = least noisy estimate for deterministic kernels
CPU=${CPU:-}                  # e.g. CPU=2 pins every measurement to one (performance) core
run() { if [ -n "$CPU" ]; then taskset -c "$CPU" "$@"; else "$@"; fi; }
echo "settings: simd=$SIMD reps=$REPS stat=$STAT cpu_pin=${CPU:-none}"

echo "== 1. correctness =="; ./test | tail -5

echo "== 2. primitives x levels x data size, 64 PEs, 1D (paper Figs. 14/16/18) =="
run ./bench --shape 64 --mask 1 --sizes 512,2048,8192,32768,65536 --simd $SIMD --reps $REPS --stat $STAT --csv results/sizes_1d64.csv

echo "== 3. number of PEs (paper Fig. 19) =="
first=1
for n in 8 16 32 64; do
  H=""; [ $first -eq 0 ] && H="--no-header"; first=0
  run ./bench --shape $n --mask 1 --sizes 8192,65536 --prims aa,rs,ag,ar --levels 0,3 --simd $SIMD --reps $REPS --stat $STAT $H --csv results/pe_sweep.csv
done

echo "== 4. hypercube shapes / communication axes (paper Fig. 20) =="
first=1
for cfg in "8x8 10" "8x8 01" "8x8 11" "16x4 10" "16x4 11" "4x16 01" "8x2x4 101" "8x2x4 100"; do
  set -- $cfg; H=""; [ $first -eq 0 ] && H="--no-header"; first=0
  run ./bench --shape $1 --mask $2 --sizes 32768 --prims aa,rs,ag,ar --levels 0,3 --simd $SIMD --reps $REPS --stat $STAT $H --csv results/shapes.csv
done

echo "== 5. data types (paper Sec. VIII-F) =="
first=1
for dt in 1 4 8; do
  H=""; [ $first -eq 0 ] && H="--no-header"; first=0
  run ./bench --shape 8x8 --mask 11 --sizes 32768 --prims rs,ar --levels 0,2,3 --dtype $dt --simd $SIMD --reps $REPS --stat $STAT $H --csv results/dtypes.csv
done

echo "== 6. [OUR ADDITION] SIMD flavour; '-ptr' = original function-pointer design, plain = host loops compiled per ISA =="
first=1
for sm in scalar avx2-ptr avx2 avx512-ptr avx512; do
  H=""; [ $first -eq 0 ] && H="--no-header"
  if run ./bench --shape 64 --mask 1 --sizes 32768 --prims aa,rs,ag,ar --levels 0,3 --simd $sm --reps $REPS --stat $STAT $H --csv results/simd.csv 2>/dev/null; then first=0; else echo "   ($sm not supported on this CPU - skipped)"; fi
done

echo "== 7. [OUR ADDITION] domain-agnostic reductions: OR/XOR need no domain transfer =="
run ./bench --shape 8x8 --mask 11 --sizes 32768 --prims rs,ar --levels 0,2,3 --dtype 8 --op sum --simd $SIMD --reps $REPS --stat $STAT --csv results/ops.csv
run ./bench --shape 8x8 --mask 11 --sizes 32768 --prims rs,ar --levels 0,2,3 --dtype 8 --op or  --simd $SIMD --reps $REPS --stat $STAT --no-header --csv results/ops.csv

echo "== 8. application: BFS (AllReduce with bitwise OR) =="
first=1
for n in 8 16 32 64; do
  H=""; [ $first -eq 0 ] && H="--no-header"; first=0
  run ./bfs --n $n --V 16384 --deg 8 --runs 15 --stat $STAT --simd $SIMD $H --csv results/bfs.csv
done
echo "done -> results/*.csv   (now: python3 scripts/plot.py)"
