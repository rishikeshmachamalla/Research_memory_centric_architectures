import numpy as np
import pytest
from pidcomm_sim.virtual_hypercube import VirtualHypercube
from pidcomm_sim import collectives_ref as ref

RNG = np.random.default_rng(0)
CASES = [((8,), "1"), ((16,), "1"), ((64,), "1"), ((8, 8), "10"), ((8, 8), "01"), ((8, 8), "11"),
         ((16, 4), "10"), ((8, 2, 4), "101"), ((8, 2, 4), "111"), ((8, 3), "10"), ((24,), "1")]
DTYPES = [np.uint8, np.uint32, np.uint64]


def rand(n_pe, n_elems, dtype):
    return RNG.integers(0, np.iinfo(dtype).max, size=(n_pe, n_elems), dtype=dtype, endpoint=True)


def per_elems(G, dtype, k=2):
    return G * k * 8 // np.dtype(dtype).itemsize          # bytes per PE = G*8*k


# ---- hand-computed examples (8 PEs, 1 group) ------------------------------------------------
def test_alltoall_hand_example():
    cube = VirtualHypercube((8,))
    data = (np.arange(8)[:, None] * 10 + np.arange(8)[None, :]).astype(np.uint64)   # data[p][j] = 10p + j
    out = ref.all_to_all(data, cube, "1")
    assert out[3].tolist() == [10 * i + 3 for i in range(8)]                          # PE3 gets chunk 3 of everyone


def test_allgather_hand_example():
    cube = VirtualHypercube((8,))
    data = np.arange(8, dtype=np.uint64).reshape(8, 1)
    out = ref.all_gather(data, cube, "1")
    assert out.shape == (8, 8) and all(out[p].tolist() == list(range(8)) for p in range(8))


def test_reducescatter_and_allreduce_hand_example():
    cube = VirtualHypercube((8,))
    data = np.tile(np.arange(8, dtype=np.uint64), (8, 1)) + np.arange(8, dtype=np.uint64)[:, None]  # data[p][e] = p + e
    rs = ref.reduce_scatter(data, cube, "1")
    assert rs.shape == (8, 1) and rs[:, 0].tolist() == [sum(p + e for p in range(8)) for e in range(8)]
    ar = ref.all_reduce(data, cube, "1")
    assert (ar == ar[0]).all() and ar[0].tolist() == rs[:, 0].tolist()


def test_integer_sum_wraps_like_hardware():
    cube = VirtualHypercube((8,))
    data = np.full((8, 64), 200, dtype=np.uint8)   # 64 bytes per PE = group size * 8
    assert (ref.all_reduce(data, cube, "1") == (8 * 200) % 256).all()


# ---- properties over many shapes/masks/dtypes -------------------------------------------------
@pytest.mark.parametrize("dims,mask", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_properties(dims, mask, dtype):
    cube = VirtualHypercube(dims)
    G = cube.group_size(mask)
    data = rand(cube.n_pe, per_elems(G, dtype), dtype)
    # AlltoAll is an involution
    assert (ref.all_to_all(ref.all_to_all(data, cube, mask), cube, mask) == data).all()
    # AllReduce == ReduceScatter followed by AllGather (the "naive" decomposition)
    for op in ("sum", "or", "xor"):
        assert (ref.all_reduce(data, cube, mask, op) ==
                ref.all_gather(ref.reduce_scatter(data, cube, mask, op), cube, mask)).all()
    # AllGather output layout
    small = data[:, : 8 // np.dtype(dtype).itemsize]
    out = ref.all_gather(small, cube, mask)
    for grp in cube.groups(mask):
        for p in grp:
            assert (out[p] == small[grp].reshape(-1)).all()


def test_multi_instance_groups_are_independent():
    cube = VirtualHypercube((8, 8))
    data = rand(64, per_elems(8, np.uint32), np.uint32)
    out = ref.all_reduce(data, cube, "10")                  # 8 independent groups along x
    for grp in cube.groups("10"):
        assert (out[grp] == np.add.reduce(data[grp], axis=0, dtype=np.uint32)).all()
    other = data.copy(); other[8:16] = 0                    # changing group 1 must not alter group 0
    assert (ref.all_reduce(other, cube, "10")[0:8] == out[0:8]).all()


# ---- constraints from the paper's appendix ---------------------------------------------------
def test_group_smaller_than_8_rejected():
    with pytest.raises(ValueError, match="at least 8"):
        ref.all_reduce(rand(32, 8, np.uint64), VirtualHypercube((4, 2, 4)), "100")


def test_size_multiple_rejected():
    with pytest.raises(ValueError, match="multiple"):
        ref.all_to_all(rand(8, 9, np.uint64), VirtualHypercube((8,)), "1")


def test_signed_dtype_rejected():
    with pytest.raises(ValueError):
        ref.all_reduce(np.zeros((8, 8), dtype=np.int32), VirtualHypercube((8,)), "1")
