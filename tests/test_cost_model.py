import numpy as np
import pytest

from pidcomm_sim import cost_model as cm
from pidcomm_sim.experiments import bench_path, run_bench

needs_bench = pytest.mark.skipif(not bench_path().exists(), reason="C++ bench not built (make in cpp_backend)")


# ---- pure-math tests (no backend needed) ------------------------------------------------------
def test_level3_non_arithmetic_has_no_domain_transfer():
    for prim in ("aa", "ag"):
        assert cm.counts(prim, 3, 64, 64, 4096)["dt"] == 0
        assert cm.counts(prim, 2, 64, 64, 4096)["dt"] > 0


def test_cross_domain_applies_to_bitwise_and_8bit_but_not_32bit_sum():
    assert cm.counts("rs", 3, 64, 64, 4096, 4, "sum")["dt"] > 0
    assert cm.counts("rs", 3, 64, 64, 4096, 4, "or")["dt"] == 0
    assert cm.counts("rs", 3, 64, 64, 4096, 1, "sum")["dt"] == 0
    assert cm.counts("rs", 2, 64, 64, 4096, 4, "or")["dt"] > 0     # level 2 never skips the DT


def test_fused_allreduce_moves_less_than_naive():
    f, nv = cm.counts("ar", 3, 64, 64, 8192), cm.counts("arnaive", 3, 64, 64, 8192)
    assert f["bus"] < nv["bus"]


def test_counts_scale_linearly_with_size_and_pes():
    a, b = cm.counts("aa", 3, 64, 64, 4096), cm.counts("aa", 3, 64, 64, 8192)
    assert b["bus"] == 2 * a["bus"]
    assert cm.counts("aa", 3, 32, 32, 4096)["bus"] * 2 == a["bus"]


@pytest.mark.parametrize("args", [("aa", 3, 12, 8, 64), ("aa", 3, 64, 4, 64), ("aa", 3, 64, 64, 100), ("xx", 3, 64, 64, 4096),
                                  ("aa", 4, 64, 64, 4096), ("arnaive", 0, 64, 64, 4096)])
def test_invalid_inputs_rejected(args):
    with pytest.raises(ValueError):
        cm.counts(*args)


def test_fit_recovers_known_coefficients_from_synthetic_data():
    true = np.array([0.002, 0.01, 0.004, 0.0015, 0.006, 0.009, 0.003])   # bus dt mem rot perm comb swords
    rows = []
    for prim in ("aa", "ag", "rs", "ar"):
        for lvl in range(4):
            for n, G in ((8, 8), (16, 16), (32, 32), (64, 64)):
                for nb in (512, 2048, 8192):
                  for dt_, op in ((4, "sum"), (1, "sum"), (8, "or")):
                    c = cm.counts(prim, lvl, n, G, nb, dt_, op)
                    rows.append(dict(prim=prim, level=lvl, n_pe=n, group=G, bytes_per_pe=nb, dtype=dt_, op=op,
                                     host_ms=float(cm.feature_vector(c) @ true), shape=str(n), mask="1"))
    m = cm.fit(rows)
    assert np.allclose(m.coef, true, rtol=1e-3, atol=1e-6)
    assert cm.evaluate(m, rows)["mape"] < 1e-3
    lc = cm.level_choice_accuracy(m, rows)
    assert lc["exact_match"] == 1.0 and lc["max_regret"] == 0.0


# ---- the key check: formulas == the C++ backend's own counters (no timing involved) ---------------------
@needs_bench
@pytest.mark.parametrize("shape,mask", [((64,), "1"), ((8, 8), "10"), ((8, 8), "01"), ((16, 4), "11"), ((8, 2, 4), "101"), ((24,), "1")])
@pytest.mark.parametrize("dtype,op", [(4, "sum"), (1, "sum"), (8, "or"), (8, "xor")])
def test_event_counts_match_cpp_counters(shape, mask, dtype, op):
    sizes = [1536, 6144] if shape == (24,) else [512, 4096]
    prims = ("aa", "ag", "rs", "ar", "arnaive")
    rows = run_bench(shape, mask, sizes, prims, dtype=dtype, op=op, reps=1)
    assert rows
    for r in rows:
        c = cm.row_counts(r)
        got = {k: r[k] for k in ("bus_rd", "bus_wr", "dt", "host_mem_blk")}
        exp = {"bus_rd": c["rd"], "bus_wr": c["wr"], "dt": c["dt"], "host_mem_blk": c["mem"]}
        assert got == exp, (r["prim"], r["level"], shape, mask, dtype, op, got, exp)
