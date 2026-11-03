import pytest
from pidcomm_sim.virtual_hypercube import VirtualHypercube


def test_paper_fig5_shape_and_groups():
    cube = VirtualHypercube((4, 2, 4))                      # Fig. 5a: 32 PEs
    assert cube.n_pe == 32
    assert cube.group_size("100") == 4 and cube.n_groups("100") == 8     # Fig. 5b: 4x2 groups of size 4
    assert cube.group_size("101") == 16 and cube.n_groups("101") == 2    # Fig. 5c: 2 groups of size 16
    assert cube.group_size("010") == 2 and cube.n_groups("010") == 16


def test_x_axis_is_fastest():
    cube = VirtualHypercube((4, 2, 4))
    assert cube.groups("100")[0] == [0, 1, 2, 3]
    assert cube.groups("010")[0] == [0, 4]                  # y neighbours are 4 apart
    assert cube.groups("001")[0] == [0, 8, 16, 24]
    assert cube.groups("101")[0][:5] == [0, 1, 2, 3, 8]     # x fastest, then z


@pytest.mark.parametrize("dims", [(8,), (16,), (64,), (8, 8), (16, 4), (8, 2, 4), (4, 2, 4), (8, 3), (24,)])
def test_every_mask_partitions_all_pes(dims):
    cube = VirtualHypercube(dims)
    for m in range(2 ** len(dims)):
        mask = format(m, f"0{len(dims)}b")
        flat = sorted(p for g in cube.groups(mask) for p in g)
        assert flat == list(range(cube.n_pe))
        assert all(len(g) == cube.group_size(mask) for g in cube.groups(mask))


def test_coords_roundtrip():
    cube = VirtualHypercube((4, 2, 4))
    for pe in range(cube.n_pe):
        assert cube.pe_id(cube.coords(pe)) == pe
    assert cube.coords(13) == (1, 1, 1)


def test_group_members_share_unselected_coordinates():
    cube = VirtualHypercube((8, 4, 2))
    for g in cube.groups("101"):
        assert len({cube.coords(p)[1] for p in g}) == 1


def test_physical_model():
    assert VirtualHypercube.chip(13) == 5 and VirtualHypercube.entangled_group(13) == 1


@pytest.mark.parametrize("dims", [(), (0, 8), (6, 8), (4, 3, 8, 2, 1)[:2], (3,), (12,)])
def test_invalid_dims(dims):
    with pytest.raises(ValueError):
        VirtualHypercube(dims)


def test_last_dimension_may_be_non_power_of_two():
    assert VirtualHypercube((8, 3)).n_pe == 24


@pytest.mark.parametrize("mask", ["1", "101", "1a", ""])
def test_invalid_mask(mask):
    with pytest.raises(ValueError):
        VirtualHypercube((8, 8)).groups(mask)
