"""Thin wrapper around the optional C++ backend (cpp_backend/pcdump).  Used ONLY for cross-checking and,
later, measured host-kernel timings.  Nothing here models UPMEM hardware."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_DEFAULT = Path(__file__).resolve().parents[2] / "cpp_backend" / "pcdump"


class BackendUnavailable(RuntimeError):
    pass


class BackendRejected(ValueError):
    """The backend refused the configuration (exit code 5)."""


def backend_path() -> Path:
    return Path(os.environ.get("PIDCOMM_BACKEND", _DEFAULT))


def run_primitive(data: np.ndarray, shape: tuple[int, ...], mask: str, prim: str, level: int,
                  op: str = "sum", simd: str = "auto") -> np.ndarray:
    """Run one primitive in the C++ backend.  data: (n_pe, n_elems) unsigned ints (uint8/32/64).
    Returns the result with the same dtype, shape (n_pe, out_elems)."""
    exe = backend_path()
    if not exe.exists():
        raise BackendUnavailable(f"{exe} not built (run `make` in cpp_backend)")
    data = np.ascontiguousarray(data)
    n_pe, n_elems = data.shape
    nbytes = n_elems * data.dtype.itemsize
    if nbytes % 8:
        raise ValueError("bytes per PE must be a multiple of 8")
    words = nbytes // 8
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.bin", Path(td) / "out.bin"
        data.view(np.uint8).tofile(fin)
        cmd = [str(exe), "--shape", "x".join(map(str, shape)), "--mask", mask, "--prim", prim, "--level", str(level),
               "--dtype", str(data.dtype.itemsize), "--op", op, "--simd", simd, "--words", str(words),
               "--in", str(fin), "--out", str(fout)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 5:
            raise BackendRejected(r.stderr.strip())
        if r.returncode != 0:
            raise RuntimeError(f"backend failed (exit {r.returncode}): {r.stderr.strip()}")
        out = np.fromfile(fout, dtype=np.uint8)
    return out.view(data.dtype).reshape(n_pe, -1)
