#!/usr/bin/env bash
# Microbenchmarks behind the "AVX2 hypothesis" section of the report.  Run from anywhere:
#     CPU=2 ./cpp_backend/microbench/run_microbench.sh      (CPU = a performance core)
set -euo pipefail
cd "$(dirname "$0")/.."                       # cpp_backend
CPU=${CPU:-2}
g++ -O3 -std=c++17 -fno-strict-aliasing -Isrc microbench/mb_rotate.cpp -o /tmp/mb_rotate
g++ -O3 -std=c++17 -fno-strict-aliasing -Isrc microbench/mb_datapath.cpp -o /tmp/mb_datapath
echo "### 1. kernel alone: function pointer vs direct call (repeat 3x)"; for i in 1 2 3; do taskset -c "$CPU" /tmp/mb_rotate | tail -4; echo; done
echo "### 2. real data path (copy in -> kernel -> copy out) (repeat 3x)"; for i in 1 2 3; do taskset -c "$CPU" /tmp/mb_datapath | tail -4; echo; done
