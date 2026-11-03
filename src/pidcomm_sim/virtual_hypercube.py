"""PID-Comm *virtual hypercube* (paper Sec. IV-B, Fig. 5) -- NOT the binary-cube graph in hypercube.py.

A virtual hypercube is a multi-dimensional grid of PEs, e.g. dims=(4, 2, 4) for 32 PEs.
  * PE id = x + d0 * (y + d1 * z + ...)          (x is the fastest-varying axis)
  * a communication *mask* has one character per axis; '1' = this axis takes part in the
    communication group (e.g. "101" = x and z, as in Fig. 5c).
  * the PEs that agree on all unselected axes form one communication group; every group runs
    its own instance of the collective at the same time (multi-instance communication).
  * inside a group, members are ordered with the first selected axis varying fastest.

Simplification (state it in the report): an entangled group is modelled as 8 consecutive PE ids
(chip = id % 8, entangled group = id // 8); the paper maps onto the physical chip/bank/rank/channel order.
"""
from __future__ import annotations

from math import prod
from typing import Sequence

ENTANGLED = 8  # PEs per entangled group (one per chip of a rank)


class VirtualHypercube:
    def __init__(self, dims: Sequence[int]):
        dims = tuple(int(d) for d in dims)
        if not dims or any(d < 1 for d in dims):
            raise ValueError("dims must be a non-empty sequence of positive integers")
        for d in dims[:-1]:
            if d & (d - 1):
                raise ValueError("all dimensions except the last must be powers of two")
        n = prod(dims)
        if n % ENTANGLED:
            raise ValueError(f"number of PEs ({n}) must be a multiple of {ENTANGLED}")
        self.dims = dims
        self.n_pe = n
        self.ndim = len(dims)

    # ---- coordinates -------------------------------------------------------------------
    def coords(self, pe: int) -> tuple[int, ...]:
        self._check_pe(pe)
        out = []
        for d in self.dims:
            out.append(pe % d)
            pe //= d
        return tuple(out)

    def pe_id(self, coords: Sequence[int]) -> int:
        if len(coords) != self.ndim:
            raise ValueError("wrong number of coordinates")
        pe, stride = 0, 1
        for c, d in zip(coords, self.dims):
            if not 0 <= c < d:
                raise ValueError("coordinate out of range")
            pe += c * stride
            stride *= d
        return pe

    # ---- physical model ----------------------------------------------------------------
    @staticmethod
    def chip(pe: int) -> int:
        return pe % ENTANGLED

    @staticmethod
    def entangled_group(pe: int) -> int:
        return pe // ENTANGLED

    # ---- communication groups ----------------------------------------------------------
    def _check_mask(self, mask: str) -> None:
        if len(mask) != self.ndim or set(mask) - {"0", "1"}:
            raise ValueError(f"mask must be {self.ndim} characters of '0'/'1', got {mask!r}")

    def group_size(self, mask: str) -> int:
        self._check_mask(mask)
        return prod(d for d, m in zip(self.dims, mask) if m == "1")

    def n_groups(self, mask: str) -> int:
        return self.n_pe // self.group_size(mask)

    def groups(self, mask: str) -> list[list[int]]:
        """groups[g][i] = PE id of member i of communication group g."""
        G = self.group_size(mask)
        out = [[0] * G for _ in range(self.n_pe // G)]
        for pe in range(self.n_pe):
            rem, gi, gs, mi, ms = pe, 0, 1, 0, 1
            for d, m in zip(self.dims, mask):
                c = rem % d
                rem //= d
                if m == "1":
                    mi += c * ms
                    ms *= d
                else:
                    gi += c * gs
                    gs *= d
            out[gi][mi] = pe
        return out

    def _check_pe(self, pe: int) -> None:
        if not 0 <= pe < self.n_pe:
            raise ValueError("PE id out of range")

    def __repr__(self) -> str:
        return f"VirtualHypercube(dims={self.dims}, n_pe={self.n_pe})"
