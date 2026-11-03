#!/usr/bin/env bash
# One command that rebuilds and runs the ENTIRE evaluation with clean measurement settings.
#   SIMD=scalar CPU=2 ./scripts/final_run.sh     # MAIN run: identical scalar kernels at every level (fair ablation vs the paper)
#   SIMD=avx2   CPU=2 ./scripts/final_run.sh     # SECOND run: our AVX2 implementation (our addition)
# CPU = a performance-core logical CPU (cat /sys/devices/cpu_core/cpus).  From the project root, venv active.
# Keep the laptop plugged in, in performance mode and idle.  About 25-30 minutes per run.
# Everything for one run lands in results/run_$SIMD (data, figures) and results/final_$SIMD (text logs for the report).
set -euo pipefail
cd "$(dirname "$0")/.."
CPU=${CPU:-2}; REPS=${REPS:-15}; SIMD=${SIMD:-scalar}; PY=${PY:-python}
OUTD=results/run_$SIMD; LOG=results/final_$SIMD
mkdir -p "$LOG" "$OUTD"
L() { tee "$LOG/$1"; }

{ echo "date: $(date -Is)"; echo "simd: $SIMD  cpu pin: $CPU  reps: $REPS  stat: min"; lscpu | grep -E "Model name"
  echo "power profile: $(powerprofilesctl get 2>/dev/null || echo unknown)"; echo "governor: $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "lines with avx512 in /proc/cpuinfo: $(grep -c avx512 /proc/cpuinfo || true)"; } | L machine.txt

echo "== build =="; (cd cpp_backend && make clean >/dev/null && make) 2>&1 | L build.txt | tail -2
echo "== C++ correctness =="; (cd cpp_backend && ./test) 2>&1 | L cpp_test.txt | tail -8
echo "== Python tests =="; $PY -m pytest -q 2>&1 | L pytest.txt | tail -3
echo "== cost model: measure + validate =="; $PY scripts/validate_model.py --quick --reps "$REPS" --stat min --cpu "$CPU" --simd "$SIMD" --out "$OUTD" 2>&1 | L validate_model.txt | tail -3
echo "== error breakdown + measurement noise floor =="; $PY scripts/diagnose_model.py --data "$OUTD/measured/dataset.csv" --noise --stat min --cpu "$CPU" --simd "$SIMD" 2>&1 | L diagnose.txt | tail -3
echo "== cost model: extra features (nested) =="; $PY scripts/explore_features.py --data "$OUTD/measured/dataset.csv" 2>&1 | L explore_features.txt | tail -9
echo "== BFS application =="; $PY scripts/run_bfs.py --stat min --runs "$REPS" --cpu "$CPU" --simd "$SIMD" --data "$OUTD/measured/dataset.csv" --out "$OUTD" 2>&1 | L bfs.txt | tail -12
echo "== paper-figure sweep =="
# NOTE: cpp_backend/results/ is a SHARED path across SIMD modes - running scalar then avx2 (or avx2 then
# scalar) overwrites it, so a stale read of the LIVE cpp_backend/results/ after a second run belongs to
# whichever SIMD mode ran last, not necessarily "scalar". ALWAYS read results/final_scalar/cpp_results/
# (archived below) for the report's Table I, never cpp_backend/results/ directly once both runs have been done.
(cd cpp_backend && rm -rf results && SIMD=$SIMD CPU=$CPU REPS=$REPS STAT=min ./scripts/run_all.sh) 2>&1 | L run_all.txt | tail -2
echo "== comparison with the paper =="; (cd cpp_backend && $PY scripts/compare_with_paper.py) 2>&1 | L paper_comparison.txt
(cd cpp_backend && $PY scripts/plot.py) 2>&1 | tail -1
rm -rf "$LOG/cpp_results"; cp -r cpp_backend/results "$LOG/cpp_results"
echo; echo "ALL DONE ($SIMD).  Text logs: $LOG/   Figures: $LOG/cpp_results/*.png  $OUTD/model/*.png  $OUTD/bfs/*.png"
