"""Word-for-word comparison of the independent Python reference against the C++ backend.
Skipped automatically if cpp_backend/pcdump has not been built."""
import numpy as np
import pytest

from pidcomm_sim import collectives_ref as ref
from pidcomm_sim.backend import BackendRejected, BackendUnavailable, backend_path, run_primitive
from pidcomm_sim.virtual_hypercube import VirtualHypercube

pytestmark = pytest.mark.skipif(not backend_path().exists(), reason="C++ backend not built (make in cpp_backend)")

RNG = np.random.default_rng(42)
CASES = [((8,), "1"), ((16,), "1"), ((64,), "1"), ((24,), "1"), ((8, 8), "10"), ((8, 8), "01"), ((8, 8), "11"),
         ((16, 4), "10"), ((8, 2, 4), "101"), ((8, 2, 4), "111"), ((4, 16), "01"), ((8, 3), "11")]
PRIMS = {"aa": ref.all_to_all, "ag": ref.all_gather, "rs": ref.reduce_scatter, "ar": ref.all_reduce}


def make_data(cube, mask, prim, dtype, k=2):
    G = cube.group_size(mask)
    nbytes = 8 * k if prim == "ag" else G * 8 * k
    n = nbytes // np.dtype(dtype).itemsize
    return RNG.integers(0, np.iinfo(dtype).max, size=(cube.n_pe, n), dtype=dtype, endpoint=True)


def expected(prim, data, cube, mask, op):
    f = PRIMS[prim]
    return f(data, cube, mask, op) if prim in ("rs", "ar") else f(data, cube, mask)


@pytest.mark.parametrize("dims,mask", CASES)
@pytest.mark.parametrize("prim", ["aa", "ag", "rs", "ar"])
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_backend_matches_reference(dims, mask, prim, level):
    cube = VirtualHypercube(dims)
    if cube.group_size(mask) < 8:
        pytest.skip("group < 8 PEs")
    for dtype in (np.uint8, np.uint32, np.uint64):
        for op in (("sum", "or", "xor") if prim in ("rs", "ar") else ("sum",)):
            data = make_data(cube, mask, prim, dtype)
            got = run_primitive(data, dims, mask, prim, level, op)
            assert got.shape == expected(prim, data, cube, mask, op).shape
            assert (got == expected(prim, data, cube, mask, op)).all(), (prim, level, dtype, op)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_fused_and_naive_allreduce_agree_with_reference(level):
    cube = VirtualHypercube((8, 8))
    data = make_data(cube, "11", "ar", np.uint32)
    want = ref.all_reduce(data, cube, "11", "sum")
    for p in ("ar", "arnaive"):
        assert (run_primitive(data, (8, 8), "11", p, level) == want).all()


@pytest.mark.parametrize("simd", ["scalar", "avx2", "avx2-ptr", "avx512", "avx512-ptr", "auto"])
def test_every_simd_flavour_matches(simd):
    cube = VirtualHypercube((8, 8))
    data = make_data(cube, "11", "rs", np.uint32)
    try:
        got = run_primitive(data, (8, 8), "11", "rs", 3, "sum", simd)
    except RuntimeError as e:
        pytest.skip(str(e))
    assert (got == ref.reduce_scatter(data, cube, "11")).all()


def test_both_reject_group_smaller_than_8():
    cube = VirtualHypercube((4, 2, 4))
    data = RNG.integers(0, 2**32, size=(32, 64), dtype=np.uint64)
    with pytest.raises(ValueError):
        ref.all_reduce(data, cube, "100")
    with pytest.raises(BackendRejected):
        run_primitive(data, (4, 2, 4), "100", "ar", 3)
