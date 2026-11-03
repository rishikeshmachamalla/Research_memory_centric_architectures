"""Analytical host-cost model for the PID-Comm collectives (our contribution).

Two layers, deliberately kept apart:
  1. EVENT COUNTS  -- closed-form formulas for how many 64-byte bus bursts, domain transfers, host-memory blocks,
                      SIMD block operations and scalar word operations each (primitive, level) performs.
                      These are exact and are checked against the C++ backend's counters (no timing involved).
  2. COST COEFFICIENTS -- time = sum_k  coef_k * count_k, with non-negative coefficients fitted on MEASURED host
                      runtimes of the C++ backend (relative-error least squares).  This is a MODELED ESTIMATE of
                      laptop host runtime.  It says nothing about UPMEM hardware time.

Levels: 0 baseline | 1 +PR | 2 +PR+IM | 3 +PR+IM+CM.   Sizes: `nbytes` is bytes per PE of the larger side
(input for AA/RS/AR, OUTPUT for AG), matching bench.cpp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("bus", "dt", "mem", "rot", "perm", "comb", "swords")
PRIMS = ("aa", "ag", "rs", "ar", "arnaive")


def needs_dt(level: int, op: str, dtype: int) -> bool:
    """Reductions need the domain transfer unless cross-domain modulation applies (level 3 and the operator is
    layout-agnostic: bitwise ops, or 8-bit sums)."""
    return not (level == 3 and (op != "sum" or dtype == 1))


def _route_k(L, B):
    """kernel ops of a block-routing pass: raw-domain rotate at level 3, host-domain lane permute at levels 1-2."""
    return (dict(rot=B, perm=0) if L == 3 else dict(rot=0, perm=B))


def _aa(L, n, G, Sw):
    B = n // 8 * Sw
    if L == 0:
        return dict(rd=B, wr=B, dt=2 * B, mem=2 * B, rot=0, perm=0, comb=0, swords=n * Sw)
    return dict(rd=B, wr=B, dt=0 if L == 3 else 2 * B, mem=2 * B if L == 1 else 0, comb=0, swords=0, **_route_k(L, B))


def _ag(L, n, G, Sw):  # Sw = OUTPUT words per PE
    neg, wc = n // 8, Sw // G
    B = neg * Sw
    if L == 0:
        return dict(rd=neg * wc, wr=B, dt=neg * (wc + Sw), mem=neg * (wc + Sw), rot=0, perm=0, comb=0, swords=n * Sw)
    return dict(rd=B, wr=B, dt=0 if L == 3 else 2 * B, mem=2 * B if L == 1 else 0, comb=0, swords=0, **_route_k(L, B))


def _rs(L, n, G, Sw, op, dtype):  # Sw = INPUT words per PE
    neg, wc = n // 8, Sw // G
    B, out = neg * Sw, neg * wc
    if L == 0:
        return dict(rd=B, wr=out, dt=B + out, mem=B + out, rot=0, perm=0, comb=0, swords=n * Sw)
    return dict(rd=B, wr=out, dt=(B + out) if needs_dt(L, op, dtype) else 0, mem=2 * B if L == 1 else 0,
                rot=B, perm=0, comb=B - out, swords=0)


def _ar(L, n, G, Sw, op, dtype):  # fused AllReduce (reduced blocks never go back to the PEs)
    neg, wc = n // 8, Sw // G
    B, out = neg * Sw, neg * wc
    if L == 0:
        return dict(rd=B, wr=B, dt=2 * B, mem=2 * B, rot=0, perm=0, comb=0, swords=n * Sw)
    dt = ((B + out) if needs_dt(L, op, dtype) else 0) + (0 if L == 3 else 2 * B)
    k = _route_k(L, B)
    return dict(rd=B, wr=B, dt=dt, mem=(4 * B if L == 1 else 0), rot=B + k["rot"], perm=k["perm"], comb=B - out, swords=0)


def counts(prim: str, level: int, n_pe: int, group: int, nbytes: int, dtype: int = 4, op: str = "sum") -> dict:
    """Event counts of one collective call.  Raises ValueError for sizes the primitive does not accept."""
    if prim not in PRIMS or level not in range(4) or dtype not in (1, 4, 8) or op not in ("sum", "or", "xor"):
        raise ValueError("bad primitive/level/dtype/op")
    if n_pe % 8 or group < 8 or nbytes % (group * 8):
        raise ValueError("need n_pe % 8 == 0, group >= 8, bytes per PE a multiple of group*8")
    Sw = nbytes // 8
    if prim == "aa":
        c = _aa(level, n_pe, group, Sw)
    elif prim == "ag":
        c = _ag(level, n_pe, group, Sw)
    elif prim == "rs":
        c = _rs(level, n_pe, group, Sw, op, dtype)
    elif prim == "ar":
        c = _ar(level, n_pe, group, Sw, op, dtype)
    else:  # naive AllReduce = ReduceScatter then AllGather through PE memory
        if level == 0:
            raise ValueError("naive AllReduce is only defined for levels 1..3")
        a, b = _rs(level, n_pe, group, Sw, op, dtype), _ag(level, n_pe, group, Sw)
        c = {k: a[k] + b[k] for k in a}
    c["bus"] = c["rd"] + c["wr"]
    return c


def feature_vector(c: dict) -> np.ndarray:
    return np.array([c[f] for f in FEATURES], dtype=float)


# ------------------------------------------------------------------------------------------------------------
# Fitting / evaluation (modeled estimate vs MEASURED host runtime)
# ------------------------------------------------------------------------------------------------------------
@dataclass
class Model:
    coef: np.ndarray  # ms per event, one per FEATURES entry

    def predict_ms(self, c: dict) -> float:
        return float(feature_vector(c) @ self.coef)

    def as_dict(self) -> dict:
        return {f: float(v) for f, v in zip(FEATURES, self.coef)}


def row_counts(r: dict) -> dict:
    return counts(r["prim"], int(r["level"]), int(r["n_pe"]), int(r["group"]), int(r["bytes_per_pe"]), int(r["dtype"]), r["op"])


def fit(rows: list[dict]) -> Model:
    """Non-negative least squares on RELATIVE error (each row is divided by its measured time)."""
    from scipy.optimize import nnls
    A = np.array([feature_vector(row_counts(r)) for r in rows])
    y = np.array([float(r["host_ms"]) for r in rows])
    coef, _ = nnls(A / y[:, None], np.ones(len(rows)))
    return Model(coef)


def evaluate(model: Model, rows: list[dict]) -> dict:
    y = np.array([float(r["host_ms"]) for r in rows])
    p = np.array([model.predict_ms(row_counts(r)) for r in rows])
    rel = np.abs(p - y) / y
    ss_res, ss_tot = np.sum((np.log(p) - np.log(y)) ** 2), np.sum((np.log(y) - np.log(y).mean()) ** 2)
    return dict(n=len(rows), mape=float(rel.mean()), median_ape=float(np.median(rel)), p90_ape=float(np.quantile(rel, 0.9)),
                r2_log=float(1 - ss_res / ss_tot) if ss_tot else float("nan"))


def _key(r):  # one "decision": same primitive / shape / size / dtype / op, different levels
    return (r["prim"], r["shape"], r["mask"], r["bytes_per_pe"], r["dtype"], r["op"])


def level_choice_accuracy(model: Model, rows: list[dict]) -> dict:
    """How often does the model pick the same fastest level as the measurement, and how much runtime is lost
    when it is wrong (regret)?"""
    groups: dict = {}
    for r in rows:
        groups.setdefault(_key(r), []).append(r)
    hit, regrets, n = 0, [], 0
    for g in groups.values():
        if len(g) < 2:
            continue
        meas = {int(r["level"]): float(r["host_ms"]) for r in g}
        pred = {int(r["level"]): model.predict_ms(row_counts(r)) for r in g}
        best_m, best_p = min(meas, key=meas.get), min(pred, key=pred.get)
        n += 1
        hit += best_m == best_p
        regrets.append(meas[best_p] / meas[best_m] - 1.0)
    return dict(decisions=n, exact_match=hit / n if n else float("nan"),
                mean_regret=float(np.mean(regrets)) if regrets else float("nan"),
                max_regret=float(np.max(regrets)) if regrets else float("nan"))


def speedup_errors(model: Model, rows: list[dict]) -> list[dict]:
    """Predicted vs measured speedup of level 3 over level 0 (the headline number of the paper's Fig. 14/16)."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(_key(r), {})[int(r["level"])] = r
    out = []
    for k, g in groups.items():
        if 0 in g and 3 in g:
            m = float(g[0]["host_ms"]) / float(g[3]["host_ms"])
            p = model.predict_ms(row_counts(g[0])) / model.predict_ms(row_counts(g[3]))
            out.append(dict(prim=k[0], shape=k[1], mask=k[2], bytes_per_pe=int(k[3]), dtype=int(k[4]), op=k[5],
                            measured=m, predicted=p, rel_err=abs(p - m) / m))
    return out
