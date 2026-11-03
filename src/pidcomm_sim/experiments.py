"""Runs the C++ `bench` program from Python and collects MEASURED host runtimes (laptop, not UPMEM)."""
from __future__ import annotations

import csv
import os
import platform
import subprocess
import tempfile
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[2] / "cpp_backend" / "bench"
INT_FIELDS = ("n_pe", "group", "bytes_per_pe", "dtype", "level", "bus_rd", "bus_wr", "dt", "host_mem_blk", "pe_words")
FLOAT_FIELDS = ("host_ms", "pe_ms", "gbps")


def bench_path() -> Path:
    return Path(os.environ.get("PIDCOMM_BENCH", _DEFAULT))


def run_bench(shape, mask, sizes, prims=("aa", "rs", "ag", "ar"), levels=(0, 1, 2, 3), dtype=4, op="sum",
              simd="auto", reps=5, stat="median") -> list[dict]:
    exe = bench_path()
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not built (run `make` in cpp_backend)")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "b.csv"
        cmd = [str(exe), "--shape", "x".join(map(str, shape)), "--mask", mask, "--sizes", ",".join(map(str, sizes)),
               "--prims", ",".join(prims), "--levels", ",".join(map(str, levels)), "--dtype", str(dtype), "--op", op,
               "--simd", simd, "--reps", str(reps), "--stat", stat, "--csv", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bench failed (exit {r.returncode}): {r.stderr.strip()}")
        rows = list(csv.DictReader(open(out)))
    for row in rows:
        for f in INT_FIELDS:
            row[f] = int(row[f])
        for f in FLOAT_FIELDS:
            row[f] = float(row[f])
    return rows


def pin_cpu(cpu: int | None) -> None:
    """Pin this process (and the benchmark subprocesses it starts) to one logical CPU - removes core-migration noise."""
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})


def performance_cores() -> str | None:
    """On hybrid Intel CPUs (P+E cores) returns the logical CPUs of the performance cores, e.g. '0-7'."""
    try:
        return open("/sys/devices/cpu_core/cpus").read().strip()
    except OSError:
        return None


def environment() -> dict:
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "python": platform.python_version(), "machine": platform.machine(), "system": platform.platform()}


# one entry = (shape, mask, dtype, op, levels-or-None, prims-or-None)
def dataset_plan(quick: bool = False):
    sizes = (512, 2048, 8192, 32768) if quick else (512, 2048, 8192, 32768, 65536)
    plan = [((n,), "1") for n in (8, 16, 32, 64)] + [((8, 8), m) for m in ("10", "01", "11")] + \
           [((16, 4), "10"), ((16, 4), "11"), ((8, 2, 4), "101")]
    jobs = [(s, m, sizes, 4, "sum") for s, m in plan]
    jobs += [((8, 8), "11", sizes, 1, "sum"), ((8, 8), "11", sizes, 8, "sum"), ((8, 8), "11", sizes, 8, "or")]
    return jobs


def build_dataset(quick=False, reps=5, simd="auto", progress=print, stat="median") -> list[dict]:
    rows = []
    for shape, mask, sizes, dtype, op in dataset_plan(quick):
        prims = ("aa", "ag", "rs", "ar", "arnaive") if (dtype, op) == (4, "sum") else ("rs", "ar")
        try:
            got = run_bench(shape, mask, [s for s in sizes], prims, dtype=dtype, op=op, simd=simd, reps=reps, stat=stat)
        except RuntimeError as e:      # sizes not divisible by this group size are skipped by bench; real errors surface
            progress(f"  skipped {shape} {mask}: {e}")
            continue
        rows += got
        progress(f"  {'x'.join(map(str, shape))} mask={mask} dtype={dtype} op={op}: {len(got)} rows")
    return rows


BFS = Path(__file__).resolve().parents[2] / "cpp_backend" / "bfs"


def bfs_path() -> Path:
    return Path(os.environ.get("PIDCOMM_BFS", BFS))


def run_bfs(n_pe: int, V: int, deg: int = 8, runs: int = 5, stat: str = "median", simd: str = "auto", graph: str | None = None) -> list[dict]:
    """Run the C++ BFS (one row per implementation level 0..3)."""
    exe = bfs_path()
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not built (run `make` in cpp_backend)")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bfs.csv"
        cmd = [str(exe), "--n", str(n_pe), "--V", str(V), "--deg", str(deg), "--runs", str(runs), "--stat", stat, "--simd", simd, "--csv", str(out)]
        if graph:
            cmd += ["--graph", str(graph)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bfs failed (exit {r.returncode}): {r.stderr.strip()}")
        rows = list(csv.DictReader(open(out)))
    for row in rows:
        for f in ("n_pe", "V", "edges", "level", "iters", "bus_rd", "bus_wr", "dt", "correct", "reached", "dist_sum"):
            row[f] = int(row[f])
        for f in ("comm_host_ms", "comm_pe_ms"):
            row[f] = float(row[f])
    return rows
