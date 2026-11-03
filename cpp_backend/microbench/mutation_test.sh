#!/usr/bin/env bash
# Shows the Python-vs-C++ cross-check is not vacuous: inject a fault into the C++ lane rotation and count failing tests.
#     (venv active)   ./cpp_backend/microbench/mutation_test.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(dirname "$HERE")"; PY=${PY:-python}
rm -rf /tmp/pc_mut && cp -r "$HERE" /tmp/pc_mut
sed -i 's/KP::rot(b, key.k); s.bus_wr(key.D/KP::rot(b, (key.k + 1) \& 7); s.bus_wr(key.D/' /tmp/pc_mut/src/collectives.hpp
grep -q "key.k + 1" /tmp/pc_mut/src/collectives.hpp || { echo "fault injection failed"; exit 1; }
(cd /tmp/pc_mut && make pcdump >/dev/null)
cd "$ROOT"
echo "with a deliberately broken rotation:"; PIDCOMM_BACKEND=/tmp/pc_mut/pcdump $PY -m pytest -q tests/test_backend_crosscheck.py 2>&1 | tail -1 || true
echo "with the real backend:";                $PY -m pytest -q tests/test_backend_crosscheck.py 2>&1 | tail -1 || true
