"""Breadth-first search on the (semantic) PIM system -- the application of paper Sec. VII-C.

Edges are partitioned across PEs; every PE holds the whole frontier/visited bitmap.  Per iteration each PE expands the
frontier over its own edges into a local next-frontier bitmap, then ALL PEs run AllReduce(bitwise OR) so that each PE learns
the global next frontier.  Communication goes through collectives_ref (the golden semantics), so this module is independent of
the C++ backend; it is used to (a) validate the application logic and (b) cross-check the C++ BFS on identical graphs.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from . import collectives_ref as ref
from .virtual_hypercube import VirtualHypercube


def random_graph(V: int, deg: int = 8, seed: int = 1) -> np.ndarray:
    """Undirected random edge list, shape (E, 2); self-loops removed."""
    rng = np.random.default_rng(seed)
    e = rng.integers(0, V, size=(V * deg // 2, 2))
    return e[e[:, 0] != e[:, 1]]


def path_graph(V: int) -> np.ndarray:
    return np.stack([np.arange(V - 1), np.arange(1, V)], axis=1)


def write_graph(path, V: int, edges: np.ndarray) -> None:
    with open(Path(path), "w") as f:
        f.write(f"{V} {len(edges)}\n")
        np.savetxt(f, edges, fmt="%d")


def reference_bfs(V: int, edges: np.ndarray, src: int = 0) -> np.ndarray:
    """Plain queue BFS; -1 = unreachable."""
    adj = [[] for _ in range(V)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    dist = np.full(V, -1, dtype=np.int64); dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1; q.append(v)
    return dist


def summary(dist: np.ndarray) -> tuple[int, int]:
    """(#reached vertices, sum of their distances) -- the two numbers the C++ BFS prints."""
    ok = dist >= 0
    return int(ok.sum()), int(dist[ok].sum())


def _pack(mask: np.ndarray) -> np.ndarray:
    """bool[V] -> uint64[V/64], bit (v % 64) of word (v // 64)."""
    return np.packbits(mask, bitorder="little").view("<u8").astype(np.uint64)


def _unpack(words: np.ndarray, V: int) -> np.ndarray:
    return np.unpackbits(words.astype("<u8").view(np.uint8), bitorder="little")[:V].astype(bool)


def bfs_on_pim(V: int, edges: np.ndarray, n_pe: int, src: int = 0) -> tuple[np.ndarray, int]:
    """BFS whose communication is AllReduce(OR) over n_pe PEs. Returns (dist, #iterations)."""
    if V % (64 * n_pe):
        raise ValueError(f"V must be a multiple of 64 * n_pe = {64 * n_pe} (AllReduce chunks must be whole 64-bit words)")
    cube = VirtualHypercube((n_pe,))
    idx = np.arange(len(edges))                         # same dealing as the C++ code: edge e -> PE e%n, reverse -> PE (e+1)%n
    per_pe = [np.concatenate([edges[idx % n_pe == p], edges[(idx + 1) % n_pe == p][:, ::-1]]) for p in range(n_pe)]
    dist = np.full(V, -1, dtype=np.int64); dist[src] = 0
    visited = np.zeros(V, bool); visited[src] = True
    frontier = visited.copy(); it = 0
    while True:
        data = np.zeros((n_pe, V // 64), dtype=np.uint64)
        for p, e in enumerate(per_pe):
            nxt = np.zeros(V, bool)
            if len(e):
                nxt[e[frontier[e[:, 0]], 1]] = True
            data[p] = _pack(nxt)
        glob = ref.all_reduce(data, cube, "1", "or")    # <-- the communication
        assert (glob == glob[0]).all()
        new = _unpack(glob[0], V) & ~visited
        it += 1
        if not new.any():
            return dist, it
        visited |= new; frontier = new; dist[new] = it
