"""Reference (golden) semantics of four PID-Comm collectives, written from their definitions with NumPy.

data has shape (n_pe, n_elems) and an unsigned dtype (uint8/uint32/uint64), so integer sums wrap like the hardware.
Every communication group of the mask runs its own independent instance (multi-instance communication).
Within a group of G members (ordered as VirtualHypercube.groups):
  AlltoAll      : member j receives chunk j of every member i, stored at position i      (n_elems % G == 0)
  AllGather     : every member receives the concatenation of all members' data           (output n_elems * G)
  ReduceScatter : member j receives the element-wise reduction over i of chunk j         (output n_elems / G)
  AllReduce     : every member receives the element-wise reduction of all members' data
Constraints taken from the paper's artifact appendix: group size >= 8 and bytes per PE a multiple of G * 8
(for AllGather: bytes per PE a multiple of 8).  These are NOT enforced by VirtualHypercube.groups().
"""
from __future__ import annotations

import numpy as np

from .virtual_hypercube import VirtualHypercube

OPS = {"sum": np.add, "or": np.bitwise_or, "xor": np.bitwise_xor}


def _reduce(op: str, x: np.ndarray) -> np.ndarray:
    """Reduce over axis 0 KEEPING the element width (NumPy would otherwise widen small unsigned ints,
    which would not wrap like the 8/32-bit hardware sums)."""
    return OPS[op].reduce(x, axis=0, dtype=x.dtype)


def _prepare(data: np.ndarray, cube: VirtualHypercube, mask: str, prim: str) -> tuple[list[list[int]], int]:
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[0] != cube.n_pe:
        raise ValueError(f"data must have shape ({cube.n_pe}, n_elems)")
    if data.dtype.kind != "u":
        raise ValueError("data must use an unsigned integer dtype")
    groups = cube.groups(mask)
    G = len(groups[0])
    if G < 8:
        raise ValueError("communication group must contain at least 8 PEs")
    nbytes = data.shape[1] * data.dtype.itemsize
    need = 8 if prim == "allgather" else G * 8
    if nbytes % need:
        raise ValueError(f"{prim}: bytes per PE ({nbytes}) must be a multiple of {need}")
    return groups, G


def all_to_all(data, cube: VirtualHypercube, mask: str) -> np.ndarray:
    data = np.asarray(data)
    groups, G = _prepare(data, cube, mask, "alltoall")
    if data.shape[1] % G:
        raise ValueError("alltoall: elements per PE must be divisible by the group size")
    out = np.empty_like(data)
    for grp in groups:
        chunks = data[grp].reshape(G, G, -1)            # [source i][chunk j][elem]
        out[grp] = chunks.transpose(1, 0, 2).reshape(G, -1)  # [dest j][source i][elem]
    return out


def all_gather(data, cube: VirtualHypercube, mask: str) -> np.ndarray:
    data = np.asarray(data)
    groups, G = _prepare(data, cube, mask, "allgather")
    out = np.empty((cube.n_pe, data.shape[1] * G), dtype=data.dtype)
    for grp in groups:
        out[grp] = data[grp].reshape(1, -1)             # broadcast the concatenation to every member
    return out


def reduce_scatter(data, cube: VirtualHypercube, mask: str, op: str = "sum") -> np.ndarray:
    data = np.asarray(data)
    groups, G = _prepare(data, cube, mask, "reducescatter")
    if data.shape[1] % G:
        raise ValueError("reducescatter: elements per PE must be divisible by the group size")
    out = np.empty((cube.n_pe, data.shape[1] // G), dtype=data.dtype)
    for grp in groups:
        out[grp] = _reduce(op, data[grp].reshape(G, G, -1))   # [chunk j][elem] -> member j
    return out


def all_reduce(data, cube: VirtualHypercube, mask: str, op: str = "sum") -> np.ndarray:
    data = np.asarray(data)
    groups, G = _prepare(data, cube, mask, "allreduce")
    out = np.empty_like(data)
    for grp in groups:
        out[grp] = _reduce(op, data[grp])
    return out
